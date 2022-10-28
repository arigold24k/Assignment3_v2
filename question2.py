import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, BisectingKMeans

# number of clusters
random_int = random.randint(3, 6)


class kmean():
    def __init__(self):
        self.basepath = os.path.dirname(__file__)
        file_loc = 'mesocyclone.csv'
        self.df = pd.read_csv(file_loc, index_col=False)
        pd.set_option('display.max_columns', None)
        pd.reset_option('max_columns')
        pd.set_option('display.max_rows', 25)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def kmeancluster(self):
        print('k mean clustering')
        unNormalized = self.df
        normalized = pd.DataFrame()

        # normalizing data columns
        for i in unNormalized.columns:
            # z-score normilization value - mean/ standard deviation
            normalized[i] = (unNormalized[i] - unNormalized[i].mean()) / unNormalized[i].std()
        print(unNormalized)
        print(normalized)
        print(unNormalized.describe())
        unNormalized_w_kmeans = unNormalized.copy()
        for i in range(3, 7):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(unNormalized)
            unNormalized_w_kmeans[f'kmeans_{i}'] = kmeans.labels_
        unNormalized_w_kmeans.to_csv(f'unNormalized_kmeans.csv')

        normalized_w_kmeans = normalized.copy()
        for i in range(3, 7):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(normalized)
            normalized_w_kmeans[f'kmeans_{i}'] = kmeans.labels_
        normalized_w_kmeans.to_csv(f'normalized_kmeans.csv')

        print(normalized)

        # plt.scatter(x=normalized_w_kmeans['class'], y=normalized_w_kmeans['meanReflectivity'],  c=normalized_w_kmeans['kmeans_3'])
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.show()

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20,4])
        for i, ax in enumerate(fig.axes, start=1):
            ax.scatter(x=unNormalized_w_kmeans['meanStrength'], y=unNormalized_w_kmeans['meanReflectivity'], c=unNormalized_w_kmeans[f'kmeans_{i + 2}'])
            ax.set_ylim(-10, 10)
            ax.set_xlim(-10, 10)
            ax.set_title(f'N Clusters(unNormalized): {i + 2}')
        plt.show()

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20,4])
        for i, ax in enumerate(fig.axes, start=1):
            ax.scatter(x=normalized_w_kmeans['meanStrength'], y=normalized_w_kmeans['meanReflectivity'], c=normalized_w_kmeans[f'kmeans_{i + 2}'])
            ax.set_ylim(-10, 10)
            ax.set_xlim(-10, 10)
            ax.set_title(f'N Clusters(normalized): {i + 2}')
        plt.show()

    def bisectingkmean(self):
        print('K mean disect cluster')
        unNormalized = self.df
        normalized = pd.DataFrame()

        # normalizing data columns
        for i in unNormalized.columns:
            # z-score normilization value - mean/ standard deviation
            normalized[i] = (unNormalized[i] - unNormalized[i].mean()) / unNormalized[i].std()

        unNormalized_w_bkmeans = unNormalized.copy()
        for i in range(3, 7):
            bkmeans = BisectingKMeans(n_clusters=i, random_state=0)
            bkmeans.fit(unNormalized)
            unNormalized_w_bkmeans[f'kmeans_{i}'] = bkmeans.labels_
        unNormalized_w_bkmeans.to_csv(os.path.join(self.basepath, f'unNormalized_bkmeans.csv'))

        normalized_w_bkmeans = normalized.copy()
        for i in range(3, 7):
            bkmeans = BisectingKMeans(n_clusters=i, random_state=0)
            bkmeans.fit(normalized)
            normalized_w_bkmeans[f'kmeans_{i}'] = bkmeans.labels_
        normalized_w_bkmeans.to_csv(os.path.join(self.basepath, f'normalized_bkmeans.csv'))

        # print(normalized)

        # plt.scatter(x=normalized_w_kmeans['class'], y=normalized_w_kmeans['meanReflectivity'],  c=normalized_w_kmeans['kmeans_3'])
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.show()

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20,4])
        for i, ax in enumerate(fig.axes, start=1):
            ax.scatter(x=unNormalized_w_bkmeans['meanStrength'], y=unNormalized_w_bkmeans['meanReflectivity'], c=unNormalized_w_bkmeans[f'kmeans_{i + 2}'])
            ax.set_ylim(-10, 10)
            ax.set_xlim(-10, 10)
            ax.set_title(f'N Clusters(unNormalized): {i + 2}')
        plt.show()

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[20,4])
        for i, ax in enumerate(fig.axes, start=1):
            ax.scatter(x=normalized_w_bkmeans['meanStrength'], y=normalized_w_bkmeans['meanReflectivity'], c=normalized_w_bkmeans[f'kmeans_{i + 2}'])
            ax.set_ylim(-10, 10)
            ax.set_xlim(-10, 10)
            ax.set_title(f'N Clusters(normalized): {i + 2}')
        plt.show()

if __name__ == '__main__':
    kmean_instance = kmean()
    kmean_instance.kmeancluster()
    kmean_instance.bisectingkmean()
