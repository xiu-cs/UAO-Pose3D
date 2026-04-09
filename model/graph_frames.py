import numpy as np

class Graph():
    """Graph model for the Human3.6M skeleton.

    Args:
        strategy (string): Graph partition strategy.
        - spatial: split neighbors into root / centripetal / centrifugal /
          symmetric / temporal groups

        layout (string): Skeleton layout definition.
        - 'hm36_gt': 17-joint Human3.6M ground-truth skeleton

        max_hop (int): Maximum graph hop distance to include.
        dilation (int): Hop dilation when building adjacency partitions.

    """

    def __init__(self,
                 layout, 
                 strategy,
                 pad=0,
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop
        self.dilation = dilation
        # In this repo pad=1 corresponds to a single-frame graph.
        self.seqlen = pad
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        # Used by the spatial partition to distinguish inward/outward edges.
        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout): 
        """
        Return each node's distance to the torso-center joint.
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        """
        Repeat a per-frame edge list across all frames in the graph.
        """
        return [((front) + i*self.num_node_each, (back)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]


    def basic_layout(self,neighbour_base, sym_base):
        """
        Build the full edge templates for the graph.

        neighbour_base: physical bone connections inside one frame
        sym_base: left-right symmetric joint pairs inside one frame

        Returns:
            self_link: identity edges for every node
            time_link: same-joint connections across adjacent frames
        """
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        """
        Construct the Human3.6M graph edges and coarse body-part groups.
        """
        if layout == 'hm36_gt':
            self.num_node_each = 17

            # Zero-based version of the 17-joint Human3.6M bone graph.
            neighbour_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
                              (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                              (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)
                              ]
                        
            # Left-right symmetric joint pairs.
            sym_base = [(6, 3), (5, 2), (4, 1), (11, 14), (12, 15), (13, 16)]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            # Body-part groups are used by some downstream graph logic.
            self.la, self.ra =[11, 12, 13], [14, 15, 16]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9, 10]
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link

            # Zero-based torso-center joint.
            self.center = 8 - 1
        else:
            raise ValueError("Unknown layout.")

    def get_adjacency(self, strategy):
        """Build the partitioned adjacency tensor used by the GCN."""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        # normalize_adjacency = normalize_digraph(adjacency)
        normalize_adjacency = normalize_XY_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_forward = np.zeros((self.num_node, self.num_node))
                a_back = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            # Split edges into semantic groups so each group gets its
                            # own adjacency channel in A.
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all:
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_forward:
                                a_forward[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_back:
                                a_back[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1: 
                        A.append(a_forward)
                        A.append(a_back)

            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Unknown strategy.")
            
def get_hop_distance(num_node, edge, max_hop=1):
    """Compute the shortest hop distance between every pair of nodes."""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    """Column-normalized directed graph adjacency."""
    Dl = np.sum(A, 0) 
    num_node = A.shape[0] 
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_XY_digraph(A):
    """Normalization used by the original implementation for this graph."""
    Dl = np.sum(A, 0)
    D2 = np.sum(A, 1)
    num_node = A.shape[0] 
    Dn = np.zeros((num_node, num_node))
    Dy = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            # Dn[i, i] = Dl[i]**(-0.5)
            # Dy[i, i] = D2[i]**(-0.5)
            Dn[i, i] = Dl[i]**(-1)
            Dy[i, i] = D2[i]**(0.5)
    # AD = np.dot(np.dot(Dn, A),Dy)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    """Symmetric normalization for an undirected graph adjacency."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__=="__main__":
    graph = Graph('hm36_gt', 'spatial', 1)
    print(graph.A.shape)
