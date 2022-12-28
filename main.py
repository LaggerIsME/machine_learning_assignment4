import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def main():
    example()


def example():
    # импортирую базу данных
    iris = datasets.load_iris()
    # np.c_ преобразует срез в столбец
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['target'])
    print(iris_df.head())
    print(iris_df.describe())

    x = iris_df.iloc[:, :-1]
    y = iris_df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,
                                                        random_state=0)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    scaler = Normalizer().fit(x_train)
    normalized_x_train = scaler.transform(x_train)
    normalized_x_test = scaler.transform(x_test)
    # цвета
    di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}  # dictionary
    before = sns.pairplot(iris_df.replace({'target': di}), hue='target')
    before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)
    # предыдущий датасет, но с тренировочными нормализованным х
    iris_df_2 = pd.DataFrame(data=np.c_[normalized_x_train, y_train],
                             columns=iris['feature_names'] + ['target'])
    after = sns.pairplot(iris_df_2.replace({'target': di}), hue='target')
    after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)
    plt.show()

def task

if __name__ == '__main__':
    main()
