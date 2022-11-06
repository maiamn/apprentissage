## IMPORTS ##
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from operator import itemgetter


#####################################################################################
###################################### OUTILS #######################################
#####################################################################################
def get_data_file(file) : 
    path = './artificial/'
    databrut = arff.loadarff(open(path+file+'.arff', 'r'))
    data = [[x[0],x[1]] for x in databrut[0]]
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
    data = get_data_file(file)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    
    # Résultats
    results=[]
    
    for e in np.arange(e_min,e_max,e_step) :
        for n in range(n_min,n_max):
            tps1=time.time()
            dbscan = cluster.DBSCAN(eps=e,min_samples=n).fit(data) 
            tps2=time.time()
            rt = round((tps2-tps1)*1000)
            labels = dbscan.labels_
            try:
                score = silhouette_score(data,labels, metric='euclidean')
            except ValueError:
                score=0
            results.append((e, n, rt, score))
    max_tuple = max(results, key=itemgetter(3))
    clusters = cluster.DBSCAN(eps=max_tuple[0],min_samples=max_tuple[1]).fit_predict(data)
    
    if(len(clusters)!=0):
        print("BRUIT :",100*(clusters.tolist().count(-1)/len(clusters)))
    else:
        print("BRUIT : Problème, liste vide")
    
    title = "Résultats du clustering sur le jeu de données " + file + " (eps="+str(max_tuple[0])+ ", min_samples="+str(max_tuple[1])+", running time=" + str(max_tuple[2]) + ", clusters="+str(max(clusters)+1)
    plot_dbscan(f0,f1,clusters,title)



#banana
plot_distances_voisin("banana", 15)
cluster_dbscan("banana", 0.03, 0.04, 0.005, 13,17)

#cluto-t5-8k
plot_distances_voisin("cluto-t5-8k", 15)
cluster_dbscan("cluto-t5-8k", 5, 9, 1, 15, 20)

# # #dense-disk-5000
plot_distances_voisin("dense-disk-3000", 25)
cluster_dbscan("dense-disk-5000", 0.5, 0.7, 0.01, 22, 27)
