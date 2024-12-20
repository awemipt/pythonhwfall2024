import numpy as np
from sklearn import datasets
from typing import Tuple
from sklearn.model_selection import train_test_split

def data_preproccessing(X: np.array, y: np.array, test_size=0.2)-> Tuple[np.array, np.array, np.array, np.array]:
    data = np.column_stack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    return X_train, y_train, X_test, y_test

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.means = np.zeros((len(self.classes), self.n_features))
        self.stds = np.zeros((len(self.classes), self.n_features))
        self.prior_probs = np.zeros(len(self.classes))
        
        for i, c in enumerate(self.classes):
            class_ = X[y == c]
            self.means[i, :] = class_.mean(axis=0)
            self.stds[i, :] = class_.std(axis=0)
            self.prior_probs[i] = len(class_) / len(X)


    @staticmethod
    def pdf(x, mean, std):
        exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exp
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            postreior_probs = []
            for j in range(len(self.classes)):
                probs = np.log(self.prior_probs[j]) + np.sum(np.log(self.pdf(x, self.means[j], self.stds[j])))
                postreior_probs.append(probs)
                
            y_pred[i] = self.classes[np.argmax(postreior_probs)]
        self.y_pred = y_pred
        return y_pred
    
    def score(self,  y):
        try:
            return np.mean(self.y_pred == y)
        except Exception as e:
            print(f"{e} \n may be you not fitted?")




def main():
    iris = datasets.load_iris()

    X, y = iris.data, iris.target
    data = np.column_stack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)
    X_train, y_train, X_test, y_test = data_preproccessing(X,y)
    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    print(f"Score = {clf.score(y_test)}")
if __name__=="__main__":
    main()
