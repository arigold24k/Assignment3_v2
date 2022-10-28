import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output




file_loc =  'mesocyclone.csv'
df = pd.read_csv(file_loc, index_col=False)
pd.set_option('display.max_columns', None)
pd.reset_option('max_columns')
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# cleaning up data
# df = df.dropna()
# normalized_data = df.copy()
# normalized_data = ((normalized_data - normalized_data.mean())/normalized_data.std())
# print(df.head())
# print(data.describe())

# print(data.head())
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
        print(centroid)
    return pd.concat(centroids, axis=1)


# centroids = random_centroids(normalized_data, 5)
#
# print(centroids)

# calculating getting distance of all points to the centroids
def get_labesl(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


# labels = get_labesl(normalized_data, centroids)
# print(labels.value_counts())

# calculating geometric mean to get get center of cluster
def new_controids(data, labels, k):
    centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)

    clear_output(wait=True)

    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

def run_kmeans(data, p_k):
    max_iteration = 1000
    k = p_k

    centroids = random_centroids(data, k)
    old_centroids = pd.DataFrame()
    iteration = 1
    # labels = ''
    while iteration < max_iteration and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labesl(data, centroids)
        centroids = new_controids(data, labels, k)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1


df = df.dropna()
normalized_data = df.copy()
normalized_data = ((normalized_data - normalized_data.mean())/normalized_data.std())

run_kmeans(normalized_data, 3)





