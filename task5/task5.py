import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
     
from collections import deque
class HNSW:
    def __init__(self, X_train, y_train, M=5, layers=3, layer_ratio=4):

        self.X = X_train
        self.y_train =y_train
        self.M = M
        self.layers = layers
        self.layer_ratio = layer_ratio
        self.graph_layers = [nx.Graph() for _ in range(layers)]
        self.build_hnsw()
    
    def build_hnsw(self):
       

        nbrs = NearestNeighbors(n_neighbors=self.M + 1, algorithm='auto').fit(self.X)
        distances, indices = nbrs.kneighbors(self.X)
        
        for idx, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # Skip the point itself
                self.graph_layers[0].add_edge(idx, neighbor)
        
        previous_layer_indices = np.arange(self.X.shape[0])
        for layer in range(1, self.layers):
           
            selected_indices = previous_layer_indices[::self.layer_ratio]
            self.graph_layers[layer] = nx.Graph()
            
            if len(selected_indices) == 0:
                continue
            
            selected_X = self.X[selected_indices]
            nbrs = NearestNeighbors(n_neighbors=self.M + 1, algorithm='auto').fit(selected_X)
            distances, indices = nbrs.kneighbors(selected_X)
            
            for i, neighbors in enumerate(indices):
                current_idx = selected_indices[i]
                for neighbor in neighbors[1:]:  
                    neighbor_idx = selected_indices[neighbor]
                    self.graph_layers[layer].add_edge(current_idx, neighbor_idx)
           
            previous_layer_indices = selected_indices

    def search(self, query, k=1):
        
        current_layer = self.layers - 1
        
        while current_layer >=0 and self.graph_layers[current_layer].number_of_nodes() == 0:
            current_layer -=1
        if current_layer <0:
            raise ValueError("HNSW has no layers with nodes.")
   
        entry_points = list(self.graph_layers[current_layer].nodes())
        if not entry_points:
            raise ValueError(f"No entry points in layer {current_layer}.")

        current = entry_points[0]
        visited = set()
        while current_layer >=0:
            neighbors = list(self.graph_layers[current_layer].neighbors(current))
            visited.update(neighbors)
         
            if neighbors:
                distances = np.linalg.norm(self.X[neighbors] - query, axis=1)
                min_idx = np.argmin(distances)
                closest = neighbors[min_idx]
                if np.linalg.norm(self.X[closest] - query) < np.linalg.norm(self.X[current] - query):
                    current = closest
            current_layer -=1

        queue = deque([current])
        visited = set([current])
        neighbors_found = []
        
        while queue and len(neighbors_found) < k:
            node = queue.popleft()
            neighbors_found.append(node)
            for neighbor in self.graph_layers[0].neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        

        if len(neighbors_found) < k:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(self.X)
            distances, indices = nbrs.kneighbors([query])
            return indices[0].tolist()
        
        return neighbors_found[:k]
    
    def predict(self, X_test, y_test):
       
 
        y_pred = []
        for query in X_test:
            neighbors = self.search(query, k=5)  
            neighbor_labels = self.y_train[neighbors]  
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            y_pred.append(most_common)
    
        accuracy = np.mean(np.array(y_pred) == y_test)
        return y_pred, accuracy
    def plot_graph(self, num_nodes=100, figsize=(12, 8)):
        
        layers_to_plot = list(range(self.layers))
        
        plt.figure(figsize=figsize)
        color_map = plt.get_cmap('viridis', self.layers)
        
        for layer in layers_to_plot:
            G = self.graph_layers[layer]
            if len(G) ==0:
                
                continue
            
            if len(G) > num_nodes:
                nodes = list(G.nodes())[:num_nodes]
                G = G.subgraph(nodes)
            pos = nx.spring_layout(G, seed=42)  
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color=[color_map(layer)]*len(G.nodes()), label=f"Layer {layer}")
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color=[color_map(layer)]*len(G.edges()))
        
        plt.title("HNSW Graph Visualization")
        plt.legend(scatterpoints=1)
        plt.axis('off')
        plt.savefig("5.jpg")

def main():
    data = load_iris()
    X = data.data
    y = data.target


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hnsw = HNSW(X_train,y_train, M=5, layers=3, layer_ratio=4)

    y_pred, accuracy = hnsw.predict(X_test, y_test)
    hnsw.plot_graph()
    print(f"{accuracy=}")


if __name__ == "__main__":
    main()