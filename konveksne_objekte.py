from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
from matplotlib.path import Path
from itertools import combinations
from typing import Union, Tuple
import random
import math


num=15
points = np.random.uniform(0, 10, size=(num, 2))  # Random points in 2-D (15 between 0 ans 10)
hull = ConvexHull(points) 

points1 = npi.difference(points,points[hull.vertices]) #tocke brez tiste na ovojnici vecje C
hull1= ConvexHull(points1) 
points2= points1[hull1.vertices] # tocke na C1 notranji

#Bounding box
bbox = [hull.min_bound, hull.max_bound]

num_hits = 0 # gre skozi obe hkrati
num_tries = 4 #koliko krat NE gre skozi obe (stevilo premic)
num_miss =0



#def magnitude(vector):
#   return np.sqrt(np.dot(np.array(vector),np.array(vector)))
#
#def norm(vector):
#   return np.array(vector)/magnitude(np.array(vector))
#

# ne dela pravilno
def lineRayIntersectionPoint(rayOrigin, rayDirection, point1, point2):
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array(rayDirection, dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)
    
    rayOrigin = np.append(1,rayOrigin)
    rayDirection = np.append(1,rayDirection)
    point1=np.append(1,point1)
    point2=np.append(1,point2)
    v1=np.linalg.det([rayOrigin,rayDirection,point1])
    v2=np.linalg.det([rayOrigin,rayDirection,point2])
    z1=np.linalg.det([point1,point2,rayOrigin])
    z2=np.linalg.det([point1,point2,rayDirection])
    if z1*z2<=0 and v1*v2<=0:
        return True
    return False

    
    
    
#
#    v1 = rayOrigin - point1
#    v2 = point2 - point1
#    v3 = np.array([-rayDirection[1], rayDirection[0]])
#    t1 = np.cross(v2, v1) / np.dot(v2, v3)
#    t2 = np.dot(v1, v3) / np.dot(v2, v3)
#    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
#        return [rayOrigin + t1 * rayDirection]
#    return []
#
#
#

n = num_tries # stevilo premic
b= np.empty((n, 2))
for i in range(n):
    #nakljucna tocka v C skozi katero grejo (vecji)
    b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
    #Preverimo ce je res v c           
    while Path(hull.points[hull.vertices] ).contains_point(b[i]) == False:
        b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])


theta=[]
for i in range(len(b)):
               theta=np.append(theta,np.random.uniform(0,2*np.pi))


for i in range(len(b)):
    r = b[i]
    d1 = (math.cos(theta[i]),math.sin(theta[i])) # v eno smer
    d2 =(-math.cos(theta[i]),-math.sin(theta[i])) # v dr smer
    
    
    for j in range(len(points2)):
        z1 =points2[j-1]
        z2 = points2[j]

        if lineRayIntersectionPoint(r,d1,z1,z2) or lineRayIntersectionPoint(r,d2,z1,z2):
            num_hits+=1
            break
        

print(num_hits)   



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
            ax.axline(b[i],(math.cos(theta[i]),math.sin(theta[i])),color='k')
                      
        # a modre b rdece
        #ax.plot(*a.T, 'bo',markersize=3)
        ax.plot(*b.T, 'ro',markersize=3)
        
        
        
    ax.set_xticks(range(11)) # velikost osi 0-10
    ax.set_yticks(range(11))  
plt.show()

