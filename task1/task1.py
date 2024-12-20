import pandas as pd 
from pandas import DataFrame, Series
from sklearn import datasets
from typing import Tuple
import numpy as np 

def task1_1(df: DataFrame) -> DataFrame:
    shuffled_df = df.sample(frac=1) 
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

def task1_6_7_8(distances: np.array, y_train: Series,y_test: Series, n=5) -> np.array:
    res = {}
    for test_sample, y_test_index in zip(distances, y_test.index):
        indices = np.argpartition(test_sample, n)[:n]
        sorted_indices = indices[np.argsort(test_sample[indices])]
        res.update({y_test_index: y_train.iloc[sorted_indices].mode().values[0]})
    return Series(res)
        
def task1_10_11(y_pred: Series, y_test:Series):
    pass

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


    X_train, y_train, X_test, y_test = task1_3(train, test)
    print(y_test)
    task1_4_5res = task1_4_5(X_test, X_train)
    tasks_res.append(task1_4_5)

    task1_6_7_8_res = task1_6_7_8(task1_4_5res, y_train, y_test)
    
if __name__ == "__main__":
    main()