import numpy as np
import json
import faiss
from tqdm import tqdm
import pandas as pd

class ComputeClusters():
    '''
    Compute the cluster given all the sentence embeddings corresponding to a period of time
    '''
    def __init__(self,embeddings,indexes,device = 0):
        #embeddings = list(data.values())[0]['embeddings']
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.indexes = indexes
        self.dim = len(embeddings[0])
        self.optimal_number_clusters = None
        self.device = device
        self.predictions = None
        self.tweet_id_cluster = pd.DataFrame()
        self.tweet_id_cluster['tweet_id'] = self.indexes

    def compute(self):
        '''
        Method that runs all the processes for computing the optimal number of clusters and
        computes the cluster_id for each sentence embedding
        '''
        # First, compute the optimal number of clusters
        self._compute_number_clusters()
        # Secondly, compute the labels for each sentence embedding with the optimal number of clusters
        self.predictions = self._compute_clusters(self.optimal_number_clusters, prediction = True)
        self.tweet_id_cluster['cluster_id'] = self.predictions
        return self.tweet_id_cluster

    def _compute_clusters(self,number_clusters, prediction = False):
        '''
        Compute the clusters given a fixed number od clusters
        '''
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(self.dim)
        if self.device != 'cpu':
            index_flat = faiss.index_cpu_to_gpu(res,self.device,index_flat)
        
        kmeans = faiss.Clustering(self.dim,number_clusters)
        kmeans.train(self.embeddings,index_flat)
        dist, pred_labels = index_flat.search(self.embeddings,1)

        if prediction:
            predictions = pred_labels.squeeze()
            return predictions
        else:
            return (number_clusters,sum(dist))

    def _compute_number_clusters(self):
        '''
        Computes the optimal number of clusters for the given data
        '''
        # 1.- Compute the sum squared disctances of each embedding to their closest center
        K = range(5,16)
        results_clusters = {}
        print('Computing optimal number of clusters')
        for k in tqdm(K):
            sample_result = self._compute_clusters(k)
            results_clusters[sample_result[0]] = sample_result[1]

        # Compute the slope between two points
        compute_m = lambda p1,p2: (p2[1] - p1[1]) / (p2[0] - p1[0])

        # 2.- Compute the slopes in each side of the points:
            # the points are seem as the representation in the xy plane of the
            # number of clusters(x) and squared distances(y)
        m = [0]
        clusters = list(results_clusters.keys())
        for i in range(len(clusters)):
            if i+1 == len(clusters):
                m.append(0)
            else:
                k1 = clusters[i]
                p1 = (k1,results_clusters[k1])
                k2 = clusters[i+1]
                p2 = (k2,results_clusters[k2])
                m.append(abs(compute_m(p1,p2)))
        # correct the m vector
        m[0],m[-1] = m[1],m[-2]

        point_elbowness = {}
        for i in range(len(clusters)):
            point_elbowness[clusters[i]] = abs(m[i+1] - m[i])

        self.optimal_number_clusters = max(point_elbowness, key=point_elbowness.get)
        print("Optimal number of clusters: ", self.optimal_number_clusters)
        return None

    @classmethod
    def load_sentence_embeddings(cls,file):
        '''
        Load a json file with the sentence embeddings, it should receive the file path
        '''
        with open(file, 'r') as fp:
            data = json.load(fp)
        return cls(data)
