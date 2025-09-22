import sys

sys.path.append("..")
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from model.graph_frames import Graph


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj  # 4,17,17
        self.kernel_size = adj.size(0)
        #
        self.conv1d = nn.Conv1d(
            in_channels, out_channels * self.kernel_size, kernel_size=1
        )

    def forward(self, x):  # b,j,c
        # conv1d
        x = rearrange(x, "b j c -> b c j")
        x = self.conv1d(x)  # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x, "b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc // self.kernel_size, t, v)  # b,k, kc/k, 1, j
        x = torch.einsum("bkctv, kvw->bctw", (x, self.adj))  # bctw    b,c,1,j
        # x.shape b,c,1,j   [128,512,17,1]
        x = x.contiguous()
        x = rearrange(x, "b c 1 j -> b j c")
        return x.contiguous()


class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out), nn.GELU(), nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class spatial_uMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):  # zheli
        super().__init__()

        self.linear_down = linear_block(in_features, hidden_features, drop)
        self.linear_mid = linear_block(hidden_features, hidden_features, drop)
        self.linear_up = linear_block(hidden_features, in_features, drop)

    def forward(self, x):
        # down
        x_down = self.linear_down(x)
        # mid
        x_mid = self.linear_mid(x_down) + x_down
        # up1
        x = self.linear_up(x_mid)
        return x


class spatial_uMLP_Multi(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):  # zheli
        super().__init__()

        self.linear_up1 = linear_block(in_features, hidden_features // 2, drop)
        self.linear_up2 = linear_block(hidden_features // 2, hidden_features, drop)
        self.linear_mid = linear_block(hidden_features, hidden_features, drop)
        self.linear_down1 = linear_block(hidden_features, hidden_features // 2, drop)
        self.linear_down2 = linear_block(hidden_features // 2, out_features, drop)

    def forward(self, x):
        # up1
        x = self.linear_up1(x)
        # up2
        x_up2 = self.linear_up2(x)
        # mid
        x_mid = self.linear_mid(x_up2) + x_up2  # 17*N
        x_down1 = self.linear_down1(x_mid)  # 17*N//2
        # down
        x_down2 = self.linear_down2(x_down1)  # 17
        return x_down2


class channel_uMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):  # zheli
        super().__init__()

        self.linear_down = linear_block(in_features, hidden_features, drop)
        self.linear_mid = linear_block(hidden_features, hidden_features, drop)
        self.linear_up = linear_block(hidden_features, in_features, drop)

    def forward(self, x):
        # down
        x = self.linear_down(x)
        res_mid = x
        # mid
        x = self.linear_mid(x)
        x = x + res_mid
        # up
        x = self.linear_up(x)
        return x


class spatial_channel_uMLP(nn.Module):
    def __init__(
        self, dim=512, length=17, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.0
    ):  # zheli
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.channel_umlp = channel_uMLP(
            in_features=dim, hidden_features=128, act_layer=act_layer, drop=0.2
        )

        self.norm2 = norm_layer(length)
        self.spatial_umlp = spatial_uMLP(
            in_features=17, hidden_features=17 * 4, act_layer=act_layer, drop=0.15
        )

    def forward(self, x):
        res = x  # b,j,c
        x2 = self.norm1(x)
        x_s = rearrange(x, "b j c -> b c j").contiguous()
        x_s = self.norm2(x_s)
        x_s = self.spatial_umlp(x_s)
        x_s = rearrange(x_s, "b c j -> b j c").contiguous()
        x = res + self.channel_umlp(x2) + x_s
        return x


class Block(nn.Module):  #
    def __init__(
        self,
        length,
        dim,
        tokens_dim,
        channels_dim,
        adj,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        # GCN
        self.norm1 = norm_layer(length)
        self.gcn1 = GCN(dim, dim, adj)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # channel MLP
        self.norm2 = norm_layer(dim)
        self.channel_mlp = channel_uMLP(
            in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.1
        )
        # spatial MLP
        self.norm3 = norm_layer(length)
        self.spatial_mlp = spatial_uMLP_Multi(
            in_features=17,
            hidden_features=17 * 6,
            out_features=17,
            act_layer=act_layer,
            drop=0.05,
        )  # 0.15
        # self.mlp = Mlp(in_features=dim, hidden_features=1024, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # B,J,dim
        res1 = x  # b,j,c
        x_gcn_1 = x.clone()
        # gcn
        x_gcn_1 = rearrange(x_gcn_1, "b j c -> b c j").contiguous()
        x_gcn_1 = self.norm1(x_gcn_1)  # b,c,j
        x_gcn_1 = rearrange(x_gcn_1, "b j c -> b c j").contiguous()
        x_gcn_1 = self.gcn1(x_gcn_1)  # b,j,c
        x = res1 + self.drop_path(x_gcn_1)
        # uMLP channel + spatial
        res2 = x  # b,j,c
        x2 = self.norm2(x.clone())
        x_s = rearrange(x.clone(), "b j c -> b c j").contiguous()
        x_s = self.norm3(x_s)
        x_s = self.spatial_mlp(x_s)
        x_s = rearrange(x_s, "b c j -> b j c").contiguous()
        x = res2 + self.drop_path(self.channel_mlp(x2) + x_s)

        return x


class GCN_MLP(nn.Module):
    def __init__(
        self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.10, length=27
    ):
        super().__init__()
        # depth = args.layers=3, embed_dim=args.channel=512, channels_dim=args.d_hid=1024, tokens_dim=args.token_dim=256(set by myself)

        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    length,
                    embed_dim,
                    tokens_dim,
                    channels_dim,
                    adj,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm_mu = norm_layer(embed_dim)
        self.norm_s = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        mu, s = x, x
        mu, s = self.norm_mu(mu), self.norm_s(x)
        return mu, s


class encoder(nn.Module):  # 2,256,512
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        dim_0 = 2
        dim_2 = 64
        dim_output = 512

        self.fc1 = nn.Linear(dim_0, dim_2)
        self.fc5 = nn.Linear(dim_2, dim_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc5(x)
        return x


class decoder(nn.Module):  # 2,256,512
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        dim_in = in_features
        dim_hid = hidden_features
        dim_out = out_features

        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc5 = nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc5(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## GCN
        self.graph = Graph("hm36_gt", "spatial", pad=1)
        self.A = nn.Parameter(
            torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False
        ).cuda(0)
        self.encoder = encoder(2, args.channel // 2, args.channel)
        self.GCN_MLP = GCN_MLP(
            args.layers,
            args.channel,
            args.d_hid,
            args.token_dim,
            self.A,
            length=args.n_joints,
        )  # 256
        self.pred_mu = decoder(args.channel, args.channel // 2, 3)
        self.pred_s = decoder(args.channel, args.channel // 2, 1)

    def forward(self, x):
        x = rearrange(x, "b f j c -> (b f) j c").contiguous()  # B 17 2
        x = self.encoder(x)
        x_mu, x_s = self.GCN_MLP(x)
        x_mu, x_s = self.pred_mu(x_mu), self.pred_s(x_s)
        x_mu = rearrange(x_mu, "b j c -> b 1 j c").contiguous()  # B, 1, 17, 3
        x_s = rearrange(x_s, "b j c -> b 1 j c").contiguous()  # B, f=1, 17, 1
        return x_mu, x_s
