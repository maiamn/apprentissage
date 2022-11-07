## IMPORTS ##
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
import pandas as pd


#####################################################################################
###################################### OUTILS #######################################
#####################################################################################
def get_data_file(file) : 
    path = './artificial/'
    databrut = arff.loadarff(open(path+file+'.arff', 'r'))
    data = [[x[0],x[1]] for x in databrut[0]]
    return data

def get_data_file_rapport(file) : 
    path = './dataset-rapport/'
    databrut = pd.read_csv(path+file+'.txt', sep=" ", encoding = "ISO-8859-1", skipinitialspace=True)
    data = databrut.to_numpy()
    return data


def plot_graph_init(file) : 
    path = './artificial/'
    databrut = arff.loadarff(open(path+file+'.arff', 'r'))
    data = [[x[0],x[1]] for x in databrut[0]]
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    plt.scatter(f0, f1, s=8)
    plt.title("Données initiales du jeu de données " + file)
    plt.show()
    
    
#####################################################################################
############################## CLUSTERING AGGLOMERATIF ##############################
#####################################################################################
def dendrogram(file, linkage) : 
    # Données 
    data = get_data_file(file)
    linked_mat = shc.linkage(data, linkage)
    plt.figure(figsize=(12,12))
    shc.dendrogram(linked_mat, orientation='top', distance_sort='descending', show_leaf_counts=False)
    plt.show()
 
def plot_agglo(k, data, linkage, f0, f1, title) : 
    model = cluster.AgglomerativeClustering(linkage = linkage, n_clusters=k)
    model = model.fit(data)
    labels = model.labels_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(title)
    plt.show()
    
def cluster_agglo(file, k_min, k_max):
    data = get_data_file(file)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    linkages = ['single', 'ward', 'average', 'complete']
    
    for linkage in linkages : 
            
        # Valeurs de référence silhouette
        min_silhouette = 1
        best_k_silhouette = -1
        best_time_silhouette = -1
        
        # Valeurs de référence davies-bouldin
        min_davies=9999
        best_k_davies = -1
        best_time_davies = -1
        
        # Valeurs de référence calinski
        max_calinski=0
        best_k_calinski = -1
        best_time_calinski = -1
        
        for k in range(k_min, k_max) : 
            tps1 = time.time()
            agglo = cluster.AgglomerativeClustering(n_clusters=k,linkage=linkage).fit(data)
            tps2 = time.time()
            labels = agglo.labels_
            
            silhouette = silhouette_score(data, labels, metric="euclidean")
            if(abs(1-silhouette)<min_silhouette) : 
                min_silhouette = abs(1-silhouette)
                best_k_silhouette = k
                best_time_silhouette = round((tps2 - tps1)*1000)
            
            davies = davies_bouldin_score(data, labels)
            if(davies<min_davies) : 
                min_davies = davies
                best_k_davies = k 
                best_time_davies = round((tps2 - tps1)*1000)  
            
            calinski = calinski_harabasz_score(data, labels)
            if(calinski>max_calinski) : 
                max_calinski = calinski
                best_k_calinski = k
                best_time_calinski = round((tps2 - tps1)*1000)
                
        # Affichage du résultat avec Silhouette
        title = "Données après un clustering agglomératif pour : " + file + " (linkage=" + linkage + ", k=" + str(best_k_silhouette) + ")"
        plot_agglo(best_k_silhouette, data, linkage, f0, f1, title)

        # Affichage du résultat avec Davies-Bouldin
        title = "Données après un clustering agglomératif pour : " + file + " (linkage=" + linkage + ", k=" + str(best_k_davies) + ")"
        plot_agglo(best_k_davies, data, linkage, f0, f1, title)
        
        # Affichage du résultat avec Calinski-Harabasz
        title = "Données après un clustering agglomératif pour : " + file + " (linkage=" + linkage + ", k=" + str(best_k_calinski) + ")"
        plot_agglo(best_k_calinski, data, linkage, f0, f1, title)

      
# dendrogram("2d-10c", 'single')
# dendrogram("2d-10c", 'ward')
# dendrogram("2d-10c", 'average')
# dendrogram("2d-10c", 'complete')

# cluster_agglo("dartboard1", 2, 20)