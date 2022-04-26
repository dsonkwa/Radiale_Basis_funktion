# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:27:02 2021

@author: User
"""
#from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

wein = np.loadtxt('wine.data', delimiter=',')
X_data = wein[:, 1:3]
y_data = wein[:,0]

train_x, test_x, train_y, test_y = train_test_split (X_data, y_data, test_size = 0.30)

max_iters= 100;
step= 0.5


 
def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)




def kmeans(X, k, max_iters):
    """Performs k-means clustering for 1D input
        
        Arguments:
            X {ndarray} -- A Mx1 array of inputs
            k {int} -- Number of clusters
        
        Returns:
            ndarray -- A kx1 array of final cluster centers
        """
  # choisir les centres aux choix  a partir de  notre vecteur dentrer
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    print('zufällige Zentroiden', centroids)
    print('------------------------------------------------------------------------------')

    converged = False
    
    current_iter = 0

    while (not converged) and (current_iter < max_iters):
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
# cluster_list doit etre une liste (Tuple) avec la longeueur de notre centroide
        cluster_list = [[] for i in range(len(centroids))]

        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
                # on cherche la valeur minimal de notre distance_list et a cet index on ajoute 
                #le vecteur ligne x correspond
                # find the cluster that's closest to each point
            cluster_list[int(np.argmin(distances_list))].append(x)
            #print('cluster_list:', cluster_list)
            
# toute les valeur fausses (0, False) sont supprimer dans cluster_list
        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []
# on parcoure notre cluster_list qui est de longueur 3
# update clusters by taking the mean of all of the points assigned to that cluster
        for j in range(len(cluster_list)):
            
            # on calcule la moyen de chaque Cluster_liste etant donner quon a 3, 
            #axis=0 on se refaire aux lignes
            centroids.append(np.mean(cluster_list[j], axis=0))
            
        diff= np.abs(np.sum(prev_centroids) - np.sum(centroids))

        print('K-MEANS: ', int(diff))
        
        # Zuweisung von Zentroiden
        centers= np.array(centroids)
        print('new_centroids', centers)
        print('------------------------------------------------------------------------------')

#        plt.scatter(train_x[:, 0], train_x[:, 1], c = train_y, s=10, cmap = 'rainbow') 
#        plt.scatter( centers[:, 0], centers[:, 1], c = (30,150,230), s = 200, alpha = 0.9)
#        plt.show()

        
# on verifi si pattern est egal a 0, si tel est le car 
#alors converged prend la valeur true et serai actualiser a False ou bien le contraire
        converged = (diff == 0)

        current_iter += 1
        
        
    return centers, [np.std(x) for x in cluster_list]
    
    #return centers, [np.std(x) for x in cluster_list], np.array(cluster_list)



class RBF:

        def __init__(self, X, y, tX, ty, num_of_classes,
                     k, std_from_clusters=True):
            self.X = X
            self.y = y
    
            self.tX = tX
            self.ty = ty
    
            self.number_of_classes = num_of_classes
            self.k = k
            self.std_from_clusters = std_from_clusters
    
        def rbf(self, x, c, s):
            distance = get_distance(x, c)
            #return 1 / np.exp(-distance / s ** 2) 
            return  np.exp(-distance / (2*(s ** 2)))
    
        def rbf_list(self, X, centroids, std_list):
            RBF_list = []
            for x in X:
                RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
                rbf= np.array(RBF_list)
            #return np.array(RBF_list)
            return rbf
        
        
        def fit(self):
            self.centroids, self.std_list = kmeans(self.X, self.k, max_iters)
            
        # Training
        
            if not self.std_from_clusters:
                dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
                self.std_list = (np.repeat(dMax / np.sqrt(2 * self.k), self.k))*step
                
        # Aktivierung der Neroune berechnen
            RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)
            print('Aktivierungsmatrix:',RBF_X)
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')
        # Pseudoinverse
            self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T 
            print('Gewichtsmatrix', self.w)
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')
            
            #self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)
           # print('Gewichtungen:',self.w)
            
        # Predict
        def predict(self, tx):
            
            self.tX= tx
            
            RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)
            
        # y_predict berechnen
            self.pred_ty = RBF_list_tst @ self.w
            print('Vorhersage_1:',self.pred_ty)
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')
        # arrondir les valeurs predictent
            self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])
            print('Vorhersage_2_mit abrunden:',self.pred_ty)
            print('------------------------------------------------------------------------------')
            print('------------------------------------------------------------------------------')
            
            #plt.plot(train_x, train_y, '-', label= 'true')
    #        plt.plot(train_x, self.pred_ty, '-', label= 'RBF')
    #        plt.legend()
    #        plt.tight_layout()
    #        plt.show()
            #return RBF_X, self.w, RBF_list_tst, self.pred_ty, diff
            return  self.pred_ty # diff
    
    
RBF_CLASSIFIER = RBF(train_x, train_y, test_x, test_y, num_of_classes=3,
                     k=3, std_from_clusters=False)
RBF_CLASSIFIER.fit()
y_pred = RBF_CLASSIFIER.predict(test_x)

  # on calcule lerreur
diff = y_pred - test_y

print('Accuracy: %.3f' % (len(np.where(diff == 0)[0]) / len(diff)))
print('Fehler:', np.mean(np.abs(diff))) # absolut fehler 
print('-----------------------------------------------------------------')



print('------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------')

min1, max1 = train_x[:, 0].min()-1, train_x[:, 0].max()+1
min2, max2 = train_x[:, 1].min()-1, train_x[:, 1].max()+1


x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)

#reduzieren wir jedes Gitter zu einem Vektor.
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

# make predictions for the grid
grid= np.hstack((r1,r2))
y_pred_plot = RBF_CLASSIFIER.predict(grid)

 # Differenz zwischen predictRBF und Ytest feststellen

#fehler_anzahl = np.sum(y_pred != test_y)# AnzahlFehler berechnet
#print ( "Fehleranzahl = %d" % fehler_anzahl )

# reshape the predictions back into a grid
zz= y_pred_plot.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='Paired')
for class_value in range(4):
    # get row indexes for samples with this class
    row_ix = np.where(train_y == class_value)
    # create scatter of these samples
    plt.scatter(train_x[row_ix, 0], train_x[row_ix, 1], cmap='Paired')
# show the plot
plt.title ( 'RBF_Classifikation')
plt.ylabel ( 'weight_sépale ') 
plt.xlabel ( 'length sépale ') 
plt.show()