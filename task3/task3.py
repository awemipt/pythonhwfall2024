import numpy as np

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNeighborsRegressor:
    def __init__(self, neighbors=4):
        self.neighbors = neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_indices = np.argsort(distances)[:self.neighbors]
            nearest_neighbors = self.y_train[nearest_indices]
            predictions.append(nearest_neighbors.mean())
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        wape = np.sum(np.abs(y - y_pred)) / np.sum(np.abs(y)) * 100
        return wape
    
def main():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    
    knn = KNeighborsRegressor(neighbors=5)
    knn.fit(X_train_sc, y_train)
    wape = knn.score(X_test_sc, y_test)

    print(f"{wape=}")

if __name__ == "__main__":
    main()
