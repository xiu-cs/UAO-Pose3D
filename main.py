import os
import glob
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from common.arguments import opts as parse_args
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
import time
import numpy as np
import torch.nn as nn

args = parse_args().parse()
exec('from model.' + args.model + ' import Model as Gaussian')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train_gaussian(args, actions, train_loader, model, optimizer, epoch):
    return step_Gaussian('train', args, actions, train_loader, model, optimizer, epoch)

def val_gaussian(args, actions, val_loader, model):
    with torch.no_grad():
        return step_Gaussian('test', args, actions, val_loader, model)

def step_Gaussian(split, args, actions, dataLoader, model, optimizer=None, epoch=None):

    loss_all = {
                'train_loss_gaussian': AccumLoss(),
                }

    action_error_sum = define_error_list(actions)

    model_gaussian = model['gaussian']

    if split == 'train':
        model_gaussian.train()
    else:
        model_gaussian.eval()

    for i, data in enumerate(tqdm(dataLoader, 0)):

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])
        # 2d_train: [B,1,17,2] 2D_test: [B,2,1,17,2]

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0
        
        if split == 'train':
            
            mu, s = model_gaussian(input_2D)
            
            N = input_2D.size(0)

            train_loss_gaussian = gaussian_loss(mu, s, out_target)
            
            loss_all['train_loss_gaussian'].update(train_loss_gaussian.detach().cpu().numpy() * N, N)

            loss = train_loss_gaussian
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            N = input_2D.size(0)
            # test Aug
            nonflip_2D = input_2D[:, 0]
            flip_2D = input_2D[:, 1]
            
            mu_nonflip, _ = model_gaussian(nonflip_2D)
            mu_flip, _ = model_gaussian(flip_2D)
                            
            mu_flip[:, :, :, 0] *= -1
            mu_flip[:, :, args.joints_left + args.joints_right, :] = mu_flip[:, :, args.joints_right + args.joints_left, :]
            mu = (mu_nonflip + mu_flip) / 2
            
            mu[:, :, args.root_joint] = 0

            action_error_sum = test_calculation(mu, out_target, action, action_error_sum, args.dataset, subject)

    if split == 'train':
        return loss_all['train_loss_gaussian'].avg
    
    elif split == 'test':
        mu_p1, mu_p2 = print_error(args.dataset, action_error_sum, args.train)
        return mu_p1, mu_p2

def test_time_optimization_Gaussian(args, actions, dataLoader, model):

    action_error_sum = define_error_list(actions)

    model_gaussian = model['gaussian']
    model_gaussian.eval()
    split = 'test'
    print("Test time optimization, iter_num: ", args.opt_iter_num)
            
    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])
        # 2d_train: [B,1,17,2] 2D_test: [B,2,1,17,2]

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        N = input_2D.size(0)
        # test Aug
        nonflip_2D = input_2D[:, 0]
        flip_2D = input_2D[:, 1]
        
        mu_nonflip, _ = model_gaussian(nonflip_2D)
        mu_flip, _ = model_gaussian(flip_2D)
        
        if args.test_time_optimization:
            mu_nonflip_pseudo_gt = mu_nonflip
            mu_flip_pseudo_gt = mu_flip
            mu_flip_pseudo_gt[:, :, :, 0] *= -1
            mu_flip_pseudo_gt[:, :, args.joints_left + args.joints_right, :] = mu_flip_pseudo_gt[:, :, args.joints_right + args.joints_left, :]
            
            z_non_flip = nn.Parameter(nonflip_2D.clone(), requires_grad=True)
            z_flip = nn.Parameter(flip_2D.clone(), requires_grad=True)
            opt_z = torch.optim.Adam([z_non_flip, z_flip], lr = args.lr_z)

            for iter_idx in range(args.opt_iter_num):
                output_3D_nonflip, s_nonflip_gt = model_gaussian(z_non_flip)
                output_3D_flip, s_flip_gt = model_gaussian(z_flip)

                s_flip_gt[:, :, args.joints_left + args.joints_right, :] = s_flip_gt[:, :, args.joints_right + args.joints_left, :]     
                output_3D_flip[:, :, :, 0] *= -1
                output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 
                # gaussian loss for nonflip and flip
                loss_nonflip_gaussian = gaussian_loss(mu_nonflip_pseudo_gt.detach(), s_nonflip_gt.detach(), output_3D_nonflip) #
                loss_flip_gaussian = gaussian_loss(mu_flip_pseudo_gt.detach(), s_flip_gt.detach(), output_3D_flip) #

                # loss proj 2d for nonflip and flip
                output_3D_nonflip[:,:,1:] += gt_3D[:,:,:1]
                output_3D_nonflip[:,:,:1] =  gt_3D[:,:,:1]
                proj_nonflip_2d = project_to_2d(output_3D_nonflip, batch_cam)
                loss_nonflip_proj = mpjpe_cal(proj_nonflip_2d, nonflip_2D)
                output_3D_flip[:,:,1:] += gt_3D[:,:,:1] 
                output_3D_flip[:,:,:1] = gt_3D[:,:,:1]
                proj_flip_2d = project_to_2d(output_3D_flip, batch_cam)
                loss_flip_proj = mpjpe_cal(proj_flip_2d, nonflip_2D)

                opt_loss = (loss_nonflip_gaussian + loss_flip_gaussian)*args.weight_gaussian + (loss_nonflip_proj + loss_flip_proj)*args.weight_proj
                opt_z.zero_grad()
                opt_loss.backward()
                opt_z.step()
            
            with torch.no_grad():
                output_3D_nonflip_z, _ = model_gaussian(z_non_flip)
                output_3D_flip_z, _ = model_gaussian(z_flip)
            output_3D_flip_z[:, :, :, 0] *= -1
            output_3D_flip_z[:, :, args.joints_left + args.joints_right, :] = output_3D_flip_z[:, :, args.joints_right + args.joints_left, :]

            mu = (output_3D_nonflip_z + output_3D_flip_z) / 2
            
        else:
            mu_flip[:, :, :, 0] *= -1
            mu_flip[:, :, args.joints_left + args.joints_right, :] = mu_flip[:, :, args.joints_right + args.joints_left, :]
            mu = (mu_nonflip + mu_flip) / 2
        
        mu[:, :, args.root_joint] = 0

        action_error_sum = test_calculation(mu, out_target, action, action_error_sum, args.dataset, subject)

    mu_p1, mu_p2 = print_error(args.dataset, action_error_sum, args.train)
    return mu_p1, mu_p2

if __name__ == '__main__':
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime

    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = './debug/' + logtime
        elif args.train ==False:
            args.checkpoint = './test_results/' + logtime
        else:
            args.checkpoint = './checkpoint/' + logtime

        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
            
        arguments = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'arguments.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(arguments.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        # backup files
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join( args.checkpoint, args.create_time + "_" + file_name))
        shutil.copyfile(src="model/model_Gaussian.py", dst = os.path.join(args.checkpoint, args.create_time + "_model_Gaussian.py"))
        shutil.copyfile(src="common/arguments.py", dst = os.path.join(args.checkpoint, args.create_time + "_arguments.py"))
        # shutil.copyfile(src="common/utils.py", dst = os.path.join(args.checkpoint, args.create_time + "_utils.py"))
        
        if args.train:
            shutil.copyfile(src="train.sh", dst = os.path.join(args.checkpoint, args.create_time + "_train.sh"))
        elif args.test:
            shutil.copyfile(src="test.sh", dst = os.path.join(args.checkpoint, args.create_time + "_test.sh"))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'record.log'), level=logging.INFO)
        
        logging.info("Starting the record")
        logging.root.handlers[0].setFormatter(logging.Formatter('%(message)s'))

    root_path = args.root_path
    dataset_path = root_path + 'data_3d_' + args.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    if args.train:
        train_data = Fusion(opt=args, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    if args.test:
        test_data = Fusion(opt=args, train=False, dataset=dataset, root_path =root_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = {}
    model['gaussian'] = Gaussian(args).cuda()

    if args.reload_model:
        model_gaussian_dict = model['gaussian'].state_dict()
        model_path = args.model_path
        pre_dict = torch.load(model_path)
        for name, key in model_gaussian_dict.items():
            model_gaussian_dict[name] = pre_dict[name]
        model['gaussian'].load_state_dict(model_gaussian_dict)

    all_param = []
    all_paramters = 0
    lr = args.lr
    all_param += list(model['gaussian'].parameters())

    print(all_paramters)
    logging.info(all_paramters)

    optimizer = optim.Adam(all_param, lr=args.lr, amsgrad=True)
   
    starttime = datetime.datetime.now()
    best_epoch = 0

    for epoch in range(1, args.nepoch):
        if args.train:
            e_train_g = train_gaussian(args, actions, train_dataloader, model, optimizer, epoch)
            
        if args.test:
            if args.test_time_optimization:
                e_mu_p1, e_mu_p2 = test_time_optimization_Gaussian(args, actions, test_dataloader, model)
            else:
                e_mu_p1, e_mu_p2 = val_gaussian(args, actions, test_dataloader, model)
    
            data_threshold = e_mu_p1
            save_folder = os.path.join(args.checkpoint, "models")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if args.train and data_threshold < args.previous_best_threshold:
                args.previous_name = save_model(args.previous_name, save_folder, epoch, data_threshold, model['gaussian'], "Model_Gaussian_mu_p1")
                args.previous_best_threshold = data_threshold
                best_epoch = epoch
                files = glob.glob(os.path.join(args.checkpoint, "best result*"))
                for file in files:
                    os.remove(file)
                text_name = "best result:{:.3f}, best epoch:{}".format(args.previous_best_threshold, best_epoch)
                os.mknod(os.path.join(args.checkpoint, text_name))

            if args.train == 0:
                break
            else:
                print('e: %d, lr: %.7f, e_train_g: %.4f, e_mu_p1: %.2f, e_mu_p2: %.2f' % 
                (epoch, lr, e_train_g, e_mu_p1, e_mu_p2))
                logging.info('e: %d, lr: %.7f, e_train_g: %.4f, e_mu_p1: %.2f, e_mu_p2: %.2f' % 
                (epoch, lr, e_train_g, e_mu_p1, e_mu_p2))

        # if epoch % args.large_decay_epoch == 0: 
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= args.lr_decay_large
        #         lr *= args.lr_decay_large
        #         args.lr_z *= args.lr_decay_large
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= args.lr_decay
        #         lr *= args.lr_decay
        #         args.lr_z *= args.lr_decay

        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
            lr *= args.lr_decay
            args.lr_z *= args.lr_decay

    endtime = datetime.datetime.now()   
    a = (endtime - starttime).seconds
    h = a//3600
    mins = (a-3600*h)//60
    s = a-3600*h-mins*60
    print("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    logging.info("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    
    print(h,"h",mins,"mins", s,"s")
    logging.info('training time: %dh,%dmin%ds' % (h, mins,s))
    print(args.checkpoint)
    logging.info(args.checkpoint)