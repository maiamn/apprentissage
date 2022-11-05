## IMPORTS ##
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


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
    

#####################################################################################
###################################### K-MEANS ######################################
#####################################################################################
def plot_kmeans(k, data, f0, f1, title) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    labels = model.labels_
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(title)
    plt.show()
    
def cluster_kmeans(data, k_min, k_max, method) : 
    
    if method=="silhouette" : 
        # Valeurs de référence silhouette
        min_silhouette = 1
        best_k_silhouette = -1
        best_time_silhouette = -1
        
        for k in range(k_min, k_max) : 
            tps1 = time.time()
            # K-Means + plt
            model = cluster.KMeans(n_clusters=k, init='k-means++')
            model.fit(data)
            labels = model.labels_
            # Métrique silhouette
            silhouette = silhouette_score(data, labels, metric="euclidean")
            tps2 = time.time()
            
            if(abs(1-silhouette)<min_silhouette) : 
                min_silhouette = abs(1-silhouette)
                best_k_silhouette = k
                best_time_silhouette = tps2 - tps1
                
        return [best_k_silhouette, best_time_silhouette]
        
    
    elif method=="davies" : 
        # Valeurs de référence davies-bouldin
        min_davies=9999
        best_k_davies = -1
        best_time_davies = -1
        
        for k in range(k_min, k_max) : 
            tps1 = time.time()
            
            # K-Means + plt
            model = cluster.KMeans(n_clusters=k, init='k-means++')
            model.fit(data)
            labels = model.labels_
            # Métrique davies-bouldin
            davies = davies_bouldin_score(data, labels)
            tps2 = time.time()
            
            if(davies<min_davies) : 
                min_davies = davies
                best_k_davies = k 
                best_time_davies = tps2 - tps1  
                
        return [best_k_davies, best_time_davies]
        
    elif method=="calinski" : 
        # Valeurs de référence calinski
        max_calinski=0
        best_k_calinski = -1
        best_time_calinski = -1
        
        for k in range(k_min,k_max) : 
            # Temps de début de l'algorithme
            tps1 = time.time()
            # K-Means + plt
            model = cluster.KMeans(n_clusters=k, init='k-means++')
            model.fit(data)
            labels = model.labels_
            # Métrique calinski
            calinski = calinski_harabasz_score(data, labels)
            tps2 = time.time()
            
            if(calinski>max_calinski) : 
                max_calinski = calinski
                best_k_calinski = k
                best_time_calinski = tps2 - tps1
                
        return [best_k_calinski, best_time_calinski]
        
    else : 
        print("Erreur - Méthode de comparaison non définie")
        
        
def plot_res_kmeans(file, k_min, k_max) : 
    # Récupérer les données 
    data = get_data_file(file)
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    
    # Clustering K-Means
    [k_sil, time_sil] = cluster_kmeans(data, 2, 20, "silhouette")
    [k_davies, time_davies] = cluster_kmeans(data, 2, 20, "davies")
    [k_calinski, time_calinski] = cluster_kmeans(data, 2, 20, "calinski")

    # Résultats 
    print("Le meilleur k avec Silhouette est : ", k_sil, "(running time=", time_sil, ")")
    print("Le meilleur k avec Davies est : ", k_davies, "(running time=", time_davies, ")")
    print("Le meilleur k avec Calinski est : ", k_calinski, "(running time=", time_calinski, ")")


    # Affichage du résultat avec Silhouette
    title = "Données après un clustering K-Means pour : " + file + " (k=" + str(k_sil) + ", running time =" + str(time_sil) + ", method=silhouette)"
    plot_kmeans(k_sil, data, f0, f1, title)

    # Affichage du résultat avec Davies-Bouldin
    title = "Données après un clustering K-Means pour : " + file + " (k=" + str(k_davies) + ", running time =" + str(time_davies) + ", method=davies)"
    plot_kmeans(k_davies, data, f0, f1, title)
    
    # Affichage du résultat avec Calinski-Harabasz
    title = "Données après un clustering K-Means pour : " + file + " (k=" + str(k_calinski) + ", running time =" + str(time_calinski) + ", method=calinski)"
    plot_kmeans(k_calinski, data, f0, f1, title)
    
    
    
# Jeux de données pour lesquels l'algo ne fonctionne pas : banana, flame, circle, zelnik5, smile3, 3-spiral
plot_res_kmeans("2d-4c", 0, 20)
plot_res_kmeans("diamond9", 0, 20)
plot_res_kmeans("hepta", 0, 20)