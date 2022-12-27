from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
from matplotlib.path import Path

num=15
points = np.random.uniform(0, 10, size=(num, 2))  # Random points in 2-D (15 between 0 ans 10)
hull = ConvexHull(points) 

points1 = npi.difference(points,points[hull.vertices]) #tocke brez tiste na ovojnici C1
hull1= ConvexHull(points1) 

#Bounding box
bbox = [hull.min_bound, hull.max_bound]

num_hits = 0 # gre skozi obe hkrati
num_tries = 8 #koliko krat NE gre skozi obe





for plot_id in (1, 2, 3, 4, 5): 
    fig, ax = plt.subplots(ncols=1, figsize=(5, 3)) #unpacks tuple into figure and ax (Ncols=Number of columns of the subplot grid default=1 lahko bi zbrisali ,figsize=velikost kvadratka)
    
    
    if plot_id == 1: 
        ax.set_title('Dane tocke') 
        ax.plot(points[:, 0], points[:, 1], '.', color='k')# black dots (first coordinate points[:, 0] second  points[:, 1])     
    if plot_id == 2:
        ax.set_title('konveksna ovojnica konveksnega objekta C1') 
        ax.plot(points[:, 0], points[:, 1], '.', color='k')# black dots (first coordinate points[:, 0] second  points[:, 1])  
        for simplex in hull.simplices: # hull.simplicies ti da indekse 
            ax.plot(points[simplex, 0], points[simplex, 1], 'c') #narise ovojnico v cyan  
        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)#redece krogce okoli tocke ki so v ovojnici 
   
    if plot_id == 3:
        ax.plot(points[:, 0], points[:, 1], '.', color='k')# black dots (first coordinate points[:, 0] second  points[:, 1])  
        ax.set_title('Dane tocke po odstranitvi tiste iz nadobjekta C1')
        ax.plot(points1[:, 0], points1[:, 1], '.', color='r')
        
    if plot_id==4:
        
        ax.set_title('konveksni ovojnici konveksnih objektov C1 in C2')
        ax.plot(points[:, 0], points[:, 1], '.', color='k')# black dots (first coordinate points[:, 0] second  points[:, 1])   
        for simplex in hull.simplices: # hull.simplicies ti da indekse 
            ax.plot(points[simplex, 0], points[simplex, 1], 'c') #narise ovojnico v cyan  
        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)#redece krogce okoli tocke ki so v ovojnici 
        
        for simplex in hull1.simplices: # hull.simplicies ti da indekse 
            ax.plot(points1[simplex, 0], points1[simplex, 1], 'lightgreen') #narise ovojnico v cyan  
        ax.plot(points1[hull1.vertices, 0], points1[hull1.vertices, 1], 'o', mec='m', color='none', lw=1, markersize=10)
    
     
    if plot_id==5:
        
        ax.set_title('Premice') 
        for simplex in hull.simplices: # hull.simplicies ti da indekse 
            ax.plot(points[simplex, 0], points[simplex, 1], 'c') #narise ovojnico v cyan  
        
        
        for simplex in hull1.simplices: # hull.simplicies ti da indekse 
            ax.plot(points1[simplex, 0], points1[simplex, 1], 'lightgreen') #narise ovojnico v cyan  
        
        n = num_tries # stevilo premic
        b = np.empty((n, 2))
        for i in range(n):
            #nakljucna tocka v C skozi katero grejo
            b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
            #Preverimo ce je res v c           
            while Path(hull.points[hull.vertices] ).contains_point(b[i]) == False:
                b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])


        a=np.random.uniform(0, 10, size=(n, 2)) #dr tocka skozi katera grejo premice
        
        for i in range(len(b)):
          #  ax.axline(a[i],b[i], linewidth=1, color='k') 
            ax.axline(b[i],slope=np.random.uniform(-np.pi,np.pi),color='k')
        
        # a modre b rdece
        #ax.plot(*a.T, 'bo',markersize=3)
        ax.plot(*b.T, 'ro',markersize=3)
        
        
        
    ax.set_xticks(range(11)) # velikost osi 0-10
    ax.set_yticks(range(11))
plt.show()