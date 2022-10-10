## IMPORTS ##
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn import metrics
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

## PARTIE 1 ##

# Parser un fichier de donnees au format arff
# data est un tableau d’exemples avec pour chacun
# la liste des valeurs des features

# Dans les jeux de donnees consideres :
# il y a 2 features (dimension 2)
# Ex : [[−0.499261, −0.0612356] ,
# [−1.51369, 0.265446] ,
# [−1.60321, 0.362039] , . . . . .
# ]

# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster. On retire cette information
path = './artificial/'
databrut = arff.loadarff(open(path+"diamond9.arff",'r'))
data = [[x[0],x[1]] for x in databrut[0]]

# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0=[−0.499261, −1.51369 , −1.60321, . . . ]
# Ex pour f1=[−0.0612356, 0.265446, 0.362039, . . . ]

f0 = [f[0] for f in data]
f1 = [f[1] for f in data]
plt.scatter(f0,f1,s=8)
plt.title("Donnees initiales" )
plt.show()


## PARTIE 2 ##

# Les donnees sont dans data (2 dimensions)
# f0 : valeurs sur la premiere dimension
# f1 : valeurs sur la deuxieme dimension

print("Appel KMeans pour une valeur fixee de k")
tps1 = time.time()
k=3
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(data)

tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Donnees apres clustering KMeans")
plt.show()
print("nbclusters=", k, "nbiter=", iteration, "runtime=", round((tps2-tps1)*1000,2), "ms")


## PARTIE 3 ##

min_davies=9999
max_calinski=0
min_silhouette = 1
best_k_davies = -1
best_k_calinski = -1
best_k_silhouette = -1

for k in range(2,20) : 
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    
    tps2 = time.time()
    labels = model.labels_
    iteration = model.n_iter_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Donnees apres clustering KMeans")
    plt.show()
    davies = davies_bouldin_score(data, labels)
    calinski = calinski_harabasz_score(data, labels)
    silhouette = silhouette_score(data, labels, metric="euclidean")
    
    
    if(davies<min_davies) : 
        min_davies = davies
        best_k_davies = k 
        
    if(calinski>max_calinski) : 
        max_calinski = calinski
        best_k_calinski = k
        
    if(abs(1-silhouette)<min_silhouette) : 
        min_silhouette = abs(1-silhouette)
        best_k_silhouette = k
        
    print("nbclusters=", k, "nbiter=", iteration, "runtime=", round((tps2-tps1)*1000,2), "ms", "Silhouette=", silhouette, "Davies=", davies, "Calinski=", calinski)
    
print("Le meilleur k avec Davies est : ", best_k_davies, "(valeur Davies=", min_davies, ")")
print("Le meilleur k avec Calinski est : ", best_k_calinski, "(valeur Calinski=", max_calinski, ")")
print("Le meilleur k avec Silhouette est : ", best_k_silhouette, "(valeur Silhouette=", min_silhouette, ")")

## Jeux de données pour lesquels ça fonctionne : impossible, diamond9, hepta
    
## Jeux de données pour lesquels ça ne fonctionne pas : banana, flame, circle, zelnik5, smile3


## PARTIE 4 ##
min_davies=9999
max_calinski=0
min_silhouette = 1
min_kmedoid_silhouette = 1
best_k_davies = -1
best_k_calinski = -1
best_k_silhouette = -1
best_k_kmedoid_silhouette = -1

for k in range(2,20) : 
    tps1 = time.time()
    distmatrix = euclidean_distances(data)
    fp=kmedoids.fasterpam(distmatrix, k)
    tps2 = time.time()
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    print("Loss with FasterPAM:", fp.loss)
    
    plt.scatter(f0, f1, c=labels_kmed, s=8)
    plt.title("Donnees apres clustering KMdoids")
    plt.show()
    
    davies = davies_bouldin_score(data, labels_kmed)
    calinski = calinski_harabasz_score(data, labels_kmed)
    silhouette = silhouette_score(data, labels_kmed, metric="euclidean")
    
    kmedoid_silhouette = kmedoids.silhouette(distmatrix, labels_kmed)
    print(kmedoid_silhouette)
    
    if(davies<min_davies) : 
        min_davies = davies
        best_k_davies = k 
        
    if(calinski>max_calinski) : 
        max_calinski = calinski
        best_k_calinski = k
        
    if(abs(1-silhouette)<min_silhouette) : 
        min_silhouette = abs(1-silhouette)
        best_k_silhouette = k
        
    if(abs(1-kmedoid_silhouette)<min_kmedoid_silhouette) : 
        min_kmedoid_silhouette = abs(1-kmedoid_silhouette)
        best_k_kmedoid_silhouette = k
        
    print("nbclusters=", k, "nbiter=", iteration, "runtime=", round((tps2-tps1)*1000,2), "ms", "Silhouette=", silhouette, "Davies=", davies, "Calinski=", calinski, "Kmedoid silhouette=", kmedoid_silhouette)
    
print("Le meilleur k avec Davies est : ", best_k_davies, "(valeur Davies=", min_davies, ")")
print("Le meilleur k avec Calinski est : ", best_k_calinski, "(valeur Calinski=", max_calinski, ")")
print("Le meilleur k avec Silhouette est : ", best_k_silhouette, "(valeur Silhouette=", min_silhouette, ")")
print("Le meilleur k avec Kmedoid-Silhouette est : ", best_k_kmedoid_silhouette, "(valeur Silhouette=", min_kmedoid_silhouette, ")")

# Les résultats avec les calculs de scikit.learn et de kmedoids devraient être les mêmes mais il y a une 
# incohérence entre les résultats et on ne sait pas pourquoi. Les résultats données par kmedoids sont compris
# entre 0 et 1 et par scikit.learn entre -1 et 1.
