## IMPORTS ##
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from operator import itemgetter
from hdbscan import HDBSCAN
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
    
def plot_distances_voisin(file, n) : 
    # Récupérer les données
    data = get_data_file(file)
    
    # Définir les k plus proches voisins
    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title("Plus proches voisins (" + str(n) + ") pour le jeu de données " + file)
    plt.plot(trie)
    plt.show();
    
def plot_distances_voisin_rapport(file, n) : 
    # Récupérer les données
    data = get_data_file_rapport(file)
    
    # Définir les k plus proches voisins
    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(data)
    distances, indices = neigh.kneighbors(data)
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0, distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title("Plus proches voisins (" + str(n) + ") pour le jeu de données " + file)
    plt.plot(trie)
    plt.show();
    

def plot_hdbscan(f0, f1, labels, title) : 
    plt.figure(figsize=(12,12))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(title)
    plt.show()
                

    
    
    
#####################################################################################
################################ CLUSTERING HDBSCAN #################################
#####################################################################################

def cluster_hdbscan(file,n_min,n_max):
    # Récupérer les données
    data = get_data_file(file)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    
    # Résultats
    results=[]
    results_davies=[]
    results_calinski=[]
    
    for n in range(n_min,n_max):
        tps1=time.time()
        hdbscan = HDBSCAN(min_cluster_size = n).fit(data) ###########################################################
        tps2=time.time()
        rt = round((tps2-tps1)*1000)
        labels = hdbscan.labels_
        try:
            score = silhouette_score(data,labels, metric='euclidean')
            score_davies = davies_bouldin_score(data,labels)
            score_calinski = calinski_harabasz_score(data,labels)
        except ValueError:
            score=0
        results.append((n, rt, score))
        results_davies.append((n, rt, score_davies))
        results_calinski.append((n, rt, score_calinski))
    max_tuple = max(results, key=itemgetter(2))
    clusters = HDBSCAN(min_cluster_size = max_tuple[0]).fit_predict(data)
    
    min_tuple_davies = min(results_davies, key=itemgetter(2))
    clusters_davies = HDBSCAN(min_cluster_size = min_tuple_davies[0]).fit_predict(data)
    
    max_tuple_calinski = max(results_calinski, key=itemgetter(2))
    clusters_calinski = HDBSCAN(min_cluster_size = max_tuple_calinski[0]).fit_predict(data)
    
    #########################################################
    
    if(len(clusters)!=0):
        print("BRUIT :",100*(clusters.tolist().count(-1)/len(clusters)))
    else:
        print("BRUIT : Problème, liste vide")
     
    if(len(clusters_davies)!=0):
        print("BRUIT DAVIES :",100*(clusters_davies.tolist().count(-1)/len(clusters_davies)))
    else:
        print("BRUIT : Problème, liste vide")
        
    if(len(clusters_calinski)!=0):
        print("BRUIT CALINSKI :",100*(clusters_calinski.tolist().count(-1)/len(clusters_calinski)))
    else:
        print("BRUIT : Problème, liste vide")
    
    # Silhouette
    title = "Résultats du clustering Silhouette sur le jeu de données " + file + " ( min_cluster_size="+str(max_tuple[0])+", running time=" + str(max_tuple[1]) + ", clusters="+str(max(clusters)+1)+" Score = "+str(score)
    plot_hdbscan(f0,f1,clusters,title)
    
    # Davies
    title = "Résultats du clustering Davies sur le jeu de données " + file + " ( min_cluster_size="+str(min_tuple_davies[0])+", running time=" + str(min_tuple_davies[1]) + ", clusters="+str(max(clusters_davies)+1)+" Score = "+str(score_davies)
    plot_hdbscan(f0,f1,clusters_davies,title)
    
    # Calinski
    title = "Résultats du clustering Calinski sur le jeu de données " + file + " ( min_cluster_size="+str(max_tuple_calinski[0])+", running time=" + str(max_tuple_calinski[1]) + ", clusters="+str(max(clusters_calinski)+1)+" Score = "+str(score_calinski)
    plot_hdbscan(f0,f1,clusters,title)



#banana
# plot_distances_voisin("banana", 15)
# cluster_hdbscan("banana", 15,20)

# #cluto-t5-8k
# plot_distances_voisin("cluto-t5-8k", 15)
# cluster_hdbscan("cluto-t5-8k", 15, 20)

# # # #dense-disk-5000
# plot_distances_voisin("dense-disk-3000", 25)
# cluster_hdbscan("dense-disk-5000", 15, 20)

# plot_distances_voisin_rapport("x1",15)
# cluster_hdbscan("x1",15,20)

# plot_distances_voisin_rapport("x2",15)
# cluster_hdbscan("x2",15,20)

# plot_distances_voisin_rapport("x3",15)
# cluster_hdbscan("x3",15,30)

# cluster_hdbscan("x4",15,30)

# cluster_hdbscan("y1", 15,30)

# cluster_hdbscan("zz1",15,30)

# cluster_hdbscan("zz2", 15, 30)




