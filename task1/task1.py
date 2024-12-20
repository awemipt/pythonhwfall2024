import pandas as pd 
from pandas import DataFrame, Series
from sklearn import datasets
from typing import Tuple
import numpy as np 

from matplotlib import pyplot as plt
np.random.seed(10)
def task1_1(df: DataFrame) -> DataFrame:
    shuffled_df = df.sample(frac=1,random_state=1)
    return shuffled_df


def task1_2(df: DataFrame, part:float=.8) ->Tuple[DataFrame, DataFrame]:
    num_samples = df.shape[0]
    train, test = (df.iloc[:int(num_samples*part)]
            , df.iloc[int(num_samples*part):])
    return train, test


def task1_3(train: DataFrame, test: DataFrame) -> Tuple[DataFrame, Series, DataFrame, Series]:
    X_train, y_train = train.drop('target', axis=1), train['target']
    X_test, y_test = test.drop('target', axis=1), test['target']
    return X_train, y_train, X_test, y_test


def task1_4_5(X_test: DataFrame, X_train: DataFrame,) -> np.array:
    X_test_array = np.array(X_test)
    X_train_array = np.array(X_train)
    diff = X_test_array[:, np.newaxis, :] - X_train_array[np.newaxis, :, :]
    squared_diff = diff ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=2))
    return distances

def task1_6_7_8(distances: np.array, y_train: Series,X_test: Series, n=5) -> Series:
    res = {}
    for test_sample, x_test_index in zip(distances, X_test.index):
        indices = np.argsort(test_sample)
        res.update({x_test_index: y_train.iloc[indices[:n]].mode().values[0]})
        

    return Series(res)
def task1_9(y_train:Series):
    return y_train.copy()

def task1_10_11(y_pred: Series, y_test:Series):
    score = 0
    for pred, test in zip(y_pred.values, y_test.values):
        if pred==test:
            score += 1
    return score / len(y_pred)
def task1_12(df):

    train, test = task1_2(df, part=.2)
    X_train, y_train, X_test, y_test = task1_3(train, test)
    distances = task1_4_5(X_test, X_train)
    predicted = task1_6_7_8(distances, y_train, X_test)
    score = task1_10_11(predicted, y_test)
    return score
class KNN:
    def __init__(self, neighbours:int=4):
        self.neighbours = neighbours
    
    def fit(self, X_train: DataFrame, y_train: DataFrame):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        distances = task1_4_5(X_test, self.X_train)
        res = task1_6_7_8(distances, self.y_train, X_test, n=self.neighbours)
        self.res = res
        return res
    
    def score(self, y_test):
        return task1_10_11(self.res, y_test)
    

def main():
    iris = datasets.load_iris()
    tasks_res = []
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    task1_1res = task1_1(df)
    tasks_res.append(task1_1res)
    task1_2res = task1_2(task1_1res)
    train, test = task1_2res
    tast1_3res = task1_3(train, test)
    tasks_res.append(tast1_3res)


    X_train, y_train, X_test, y_test = tast1_3res

    task1_4_5res = task1_4_5(X_test, X_train)
    tasks_res.append(task1_4_5)

    task1_6_7_8res = task1_6_7_8(task1_4_5res, y_train, X_test)
    tasks_res.append(task1_6_7_8res)
    task1_10_11res = task1_10_11(task1_6_7_8res, y_test)
    print(f"aqquracy 80% train { task1_10_11res}")
    tasks_res.append(task1_10_11res)

    task1_12res = task1_12(task1_1res)

    print(f"aqquracy 20% train { task1_10_11res}")
    tasks_res.append(task1_12res)
    
    task1_19res = []
    for n in range(1, 10):
        knn = KNN(neighbours=n)
        knn.fit(X_train, y_train)
        knn.predict(X_test)
        task1_19res.append((n, knn.score(y_test)))
    x, y = zip(*task1_19res)
    print("neighbours")
    print(x)
    print("scores")
    print(y)
    plt.plot(x, y)
    plt.savefig("1.jpg")
    
if __name__ == "__main__":
    main()