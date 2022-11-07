## IMPORTS ##
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from operator import itemgetter
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
    

def plot_dbscan(f0, f1, labels, title) : 
    plt.figure(figsize=(12,12))
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(title)
    plt.show()
                

    
    
    
#####################################################################################
################################# CLUSTERING DBSCAN #################################
#####################################################################################

def cluster_dbscan(file,e_min,e_max,e_step,n_min,n_max):
    # Récupérer les données
    data = get_data_file_rapport(file)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    
    # Résultats
    results_sil=[]
    results_davies=[]
    results_calinski=[]
    
    for e in np.arange(e_min,e_max,e_step) :
        for n in range(n_min,n_max):
            tps1=time.time()
            dbscan = cluster.DBSCAN(eps=e,min_samples=n).fit(data) 
            tps2=time.time()
            rt = round((tps2-tps1)*1000)
            labels = dbscan.labels_
            try:
                score_sil = silhouette_score(data,labels, metric='euclidean')
                score_davies = davies_bouldin_score(data, labels)
                score_calinski = calinski_harabasz_score(data, labels)
            except ValueError:
                score_sil = 0
                
            results_sil.append((e, n, rt, score_sil))
            results_davies.append((e, n, rt, score_davies))
            results_calinski.append((e, n, rt, score_calinski))
                
    max_tuple_sil = max(results_sil, key=itemgetter(3))
    clusters_sil = cluster.DBSCAN(eps=max_tuple_sil[0],min_samples=max_tuple_sil[1]).fit_predict(data)
    
    min_tuple_davies = max(results_davies, key=itemgetter(3))
    clusters_davies = cluster.DBSCAN(eps=min_tuple_davies[0],min_samples=min_tuple_davies[1]).fit_predict(data)
    
    max_tuple_calinski = max(results_calinski, key=itemgetter(3))
    clusters_calinski = cluster.DBSCAN(eps=max_tuple_calinski[0],min_samples=max_tuple_calinski[1]).fit_predict(data)
    
    
    title = "Résultats du clustering sur le jeu de données " + file + " (eps="+str(max_tuple_sil[0])+ ", min_samples="+str(max_tuple_sil[1])+", running time=" + str(max_tuple_sil[2]) + ", clusters="+str(max(clusters_sil)+1) + ", score silhouette=" + str(max_tuple_sil[3]) + ")" 
    plot_dbscan(f0,f1,clusters_sil,title)
    if(len(clusters_sil)!=0):
        print("BRUIT silhouette:",100*(clusters_sil.tolist().count(-1)/len(clusters_sil)))
    else:
        print("BRUIT silhouette: Problème, liste vide")

    title = "Résultats du clustering sur le jeu de données " + file + " (eps="+str(min_tuple_davies[0])+ ", min_samples="+str(min_tuple_davies[1])+", running time=" + str(min_tuple_davies[2]) + ", clusters="+str(max(clusters_davies)+1) + ", score davies=" + str(min_tuple_davies[3]) + ")" 
    plot_dbscan(f0,f1,clusters_davies,title)
    if(len(clusters_davies)!=0):
        print("BRUIT davies:",100*(clusters_davies.tolist().count(-1)/len(clusters_davies)))
    else:
        print("BRUIT davies: Problème, liste vide")
    
    title = "Résultats du clustering sur le jeu de données " + file + " (eps="+str(max_tuple_calinski[0])+ ", min_samples="+str(max_tuple_calinski[1])+", running time=" + str(max_tuple_calinski[2]) + ", clusters="+str(max(clusters_calinski)+1) + ", score calinski=" + str(max_tuple_calinski[3]) + ")" 
    plot_dbscan(f0,f1,clusters_calinski,title)
    if(len(clusters_calinski)!=0):
        print("BRUIT calinski:",100*(clusters_calinski.tolist().count(-1)/len(clusters_calinski)))
    else:
        print("BRUIT calinski: Problème, liste vide")


# #banana
# plot_distances_voisin("banana", 15)
# cluster_dbscan("banana", 0.03, 0.04, 0.005, 13,17)

# #cluto-t5-8k
# plot_distances_voisin("cluto-t5-8k", 15)
# cluster_dbscan("cluto-t5-8k", 5, 9, 1, 15, 20)

# # # #dense-disk-5000
# plot_distances_voisin("dense-disk-3000", 25)
# cluster_dbscan("dense-disk-5000", 0.5, 0.7, 0.01, 22, 27)

# #x1
# plot_distances_voisin_rapport("x1", 15)
# cluster_dbscan("x1", 25000, 35000, 100, 13, 17)

# #x2
# plot_distances_voisin_rapport("x2", 15)
# cluster_dbscan("x2", 15000, 30000, 200, 13, 17)

# #x3
# plot_distances_voisin_rapport("x3", 15)
# cluster_dbscan("x3", 10000, 20000, 100, 13, 17)

# #x4
# plot_distances_voisin_rapport("x4", 15)
# cluster_dbscan("x4", 15000, 20000, 100, 13, 17)

# #zz1
# plot_distances_voisin_rapport("zz1", 15)
# cluster_dbscan("zz1", 1500, 4500, 50, 13, 17)

# #zz2
# plot_distances_voisin_rapport("zz2", 15)
# cluster_dbscan("zz2", 60, 90, 2, 13, 17)