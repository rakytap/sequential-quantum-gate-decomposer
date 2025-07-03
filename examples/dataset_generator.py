import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm


def grid_graph(n):
    n1 = int(np.sqrt(n))
    # assert np.sqrt(n) == n1
    print(n1, n)
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    
    for i in range(n1):
        for j in range(n1-1):
            edges.append([i*n1+j, i*n1+j+1])
    
    for i in range(n1):
        for j in range(n1-1):
            edges.append([j*n1+i, (j+1)*n1+i])
            
    G.add_edges_from(edges)
    return G


def grid8_graph(n):
    n1 = int(np.sqrt(n))
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    
    for i in range(n1):
        for j in range(n1-1):
            edges.append([i*n1+j, i*n1+j+1])
    
    for i in range(n1):
        for j in range(n1-1):
            edges.append([j*n1+i, (j+1)*n1+i])
            
    for i in range(n1-1):
        for j in range(n1-1):
            edges.append([i*n1+j, (i+1)*n1+j+1])
            
    for i in range(n1-1):
        for j in range(1, n1):
            edges.append([i*n1+j, (i+1)*n1+j-1])
      
    # this is just to construct (4,3) grid 
    if (n - n1**2) == 3:
        edges+=[[n-1, n-2], [n-1, n-4], [n-1, n-5], [n-2, n-3], [n-2, n-4],  [n-2, n-5], [n-2, n-6],
                      [n-3, n-5],  [n-3, n-6]]
       
    G.add_edges_from(edges)
    return G


def grid6_graph(n):
    n1 = int(np.sqrt(n))
    G = grid8_graph(n)
    G.remove_edges_from([[i, i+n1-1] for i in range(n)])
    return G


def grid3_graph(n):    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    
    edges.append([0,1])
    for i in range(2,n):
        edges.append([i-2, i])
        edges.append([i-1, i]) 
            
    G.add_edges_from(edges)
    return G


undirected_graphs = {
               'grid': grid_graph,
               '8grid': grid8_graph,
               '6grid': grid6_graph,
               '3grid': grid3_graph
                }


class GeneralBinaryMRF():
    def __init__(self, graph_type, n_vertices, factor_dist = "uniform", G = None):
        self.name = 'GeneralBinaryMRF'
        self.graph_type = graph_type
        if self.graph_type == "custom":
            self.graph = G
        else:
            self.graph = undirected_graphs[graph_type](n_vertices)
        self.n_vertices = self.graph.number_of_nodes()
        self.n_edges = self.graph.number_of_edges()
        self.edge_list = list(self.graph.edges())
        
        # get the cliques
        self.cliques = [np.sort(c).tolist() for c in nx.find_cliques(self.graph)]
        self.clique_sizes = [len(c) for c in self.cliques]
        # choose random factors
        if factor_dist == "uniform":
            self.factors = [np.random.random(size=(2**s)) * 100 + 1e-5 for s in self.clique_sizes]
        # from the factors calculate unnormalized measures for each assignment
        measures = []
        for i in tqdm(range(2**self.n_vertices)):
            assignment = np.array(list(bin(i)[2:].zfill(self.n_vertices))).astype(int)
            m = 1
            for j, c in enumerate(self.cliques):
                k = 0
                for l, v in enumerate(c):
                    k += assignment[v] * (2**l)
                m *= self.factors[j][k]
            measures.append(m)
            
        # get the target distribution by normalizing
        self.distribution = measures/np.sum(measures)
        
    def save(self, path):
        file = open(path+"/mrf.bin",'wb')
        pickle.dump(self,file)
        file.close()
        self.path = path
        
        
def generate_MRF_dataset(n_nodes, graph_type, dataset_size, path = None):
    mrf = GeneralBinaryMRF(graph_type, n_nodes)
    mrf_samples = np.random.choice(
        range(2**mrf.n_vertices), size=dataset_size, p=mrf.distribution
    )
    training_set = np.array(
        [
            np.array(list(format(i, "b").zfill(mrf.n_vertices))).astype(int)
            for i in mrf_samples
        ]
    )

    if path is not None:
        mrf.save(path)

    return training_set, mrf.distribution

if __name__ == "__main__":
    n_nodes = 9
    graph_type = "8grid"
    dataset_size = 1000
    saveas = "dataset.txt"
    
    training_set, target_distribution = generate_MRF_dataset(n_nodes, graph_type, dataset_size)
    
    np.savetxt(saveas, training_set.astype(int))
    
