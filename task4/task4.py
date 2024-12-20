import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self):
        self.num_column = None
        self.median = None
        self.left = None
        self.right = None
        self.arr = None

def f(node, arr, num_column=0):
    node.num_column = num_column
    
    if num_column == 4:
        node.arr = arr
        return 0
    
   
    median = np.median(arr[:, num_column])
    node.median = median
   
    left_part = arr[arr[:, num_column] <= median]
    right_part = arr[arr[:, num_column] > median]
    node.left = Node()
    f(node.left, left_part, num_column+1)
    node.right = Node()
    f(node.right, right_part, num_column+1)


class BallNode:
    def __init__(self, center=None, radius=None, left=None, right=None, points=None):
        self.center = center        
        self.radius = radius        
        self.left = left            
        self.right = right          
        self.points = points

class BallTree:
    def __init__(self, data, leaf_size=1):
        self.leaf_size = leaf_size
        self.root = self.build_tree(data)
    
    def build_tree(self, data):
        if len(data) == 0:
            return None
        center = np.mean(data, axis=0)
        
       
        distances = np.linalg.norm(data - center, axis=1)
        
       
        radius = np.max(distances)
        
       
        if len(data) <= self.leaf_size:
            return BallNode(center=center, radius=radius, points=data)
        
        idx_furthest = np.argmax(distances)
        point_furthest = data[idx_furthest]
        
       
        distances_from_furthest = np.linalg.norm(data - point_furthest, axis=1)
        idx_second_furthest = np.argmax(distances_from_furthest)
        point_second_furthest = data[idx_second_furthest]
  
  
        dist_to_furthest = np.linalg.norm(data - point_furthest, axis=1)
        dist_to_second = np.linalg.norm(data - point_second_furthest, axis=1)
        
        
        left_indices = dist_to_furthest <= dist_to_second
        right_indices = ~left_indices
        
        left_data = data[left_indices]
        right_data = data[right_indices]
        
       
        left_child = self.build_tree(left_data)
        right_child = self.build_tree(right_data)
        
        return BallNode(center=center, radius=radius, left=left_child, right=right_child)
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node is not None:
            print(f"Center: {node.center}, Radius: {node.radius}")
            if node.left or node.right:
                self.print_tree(node.left, depth + 1)
                self.print_tree(node.right, depth + 1)
            else:
                print(f"Points: {node.points}")


def main():
    iris_dataset = load_iris()
    X, y = iris_dataset.data, iris_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    print("______KDTREE_______")
    root = Node()
    f(root, X_train)
    print("Right-Right-Right-Right node arr:\n", root.right.right.right.right.arr)
    print("Right-Right-Right-Left node arr:\n", root.right.right.right.left.arr)
    print("Right-Right-Right median:", root.right.right.right.median)
    print("\n\n\n_______BALLTREE_Class_____")
    ball_tree = BallTree(X_train, leaf_size=4)  # Adjust leaf_size as needed

    ball_tree.print_tree()
    



if __name__ == "__main__":
    main()