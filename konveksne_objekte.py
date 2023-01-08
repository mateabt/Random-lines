from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
from matplotlib.path import Path
from itertools import combinations
from typing import Union, Tuple
import random
import math
from scipy.spatial.distance import euclidean
import sys

f = open('out.txt','a')
sys.stdout = f

num=15
points = np.random.uniform(0, 10, size=(num, 2))  # Random points in 2-D (15 točk med 0 in 10)
hull = ConvexHull(points) 
points1 = npi.difference(points,points[hull.vertices]) #tocke brez tistih na ovojnici vecjega objekta C
hull1= ConvexHull(points1) 
points2= points1[hull1.vertices] # tocke na C1 notranjega objekta

#Bounding box
bbox = [hull.min_bound, hull.max_bound]

num_hits = 0 # gre skozi obe hkrati
num_tries =10 #koliko krat NE gre skozi obe (stevilo premic)

print("Stevilo premic:", num_tries)
def lineRayIntersectionPoint(rayOrigin, rayDirection, point1, point2):
    rayOrigin = np.array(rayOrigin, dtype=float)
    rayDirection = np.array(rayDirection, dtype=float)
    point1 = np.array(point1, dtype=np.float64)
    point2 = np.array(point2, dtype=np.float64)
    
    rayOrigin = np.append(1,rayOrigin)
    rayDirection = np.append(1,rayDirection)
    point1=np.append(1,point1)
    point2=np.append(1,point2)
    v1=np.linalg.det([rayOrigin,rayDirection,point1])
    v2=np.linalg.det([rayOrigin,rayDirection,point2])
    z1=np.linalg.det([point1,point2,rayOrigin])
    z2=np.linalg.det([point1,point2,rayDirection])
    return (v1*v2<=0) & (z1*z2<=0)
   

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
    d1 = (10* math.cos(theta[i]),10*math.sin(theta[i])) 
    d2 = (-10* math.cos(theta[i]),-10*math.sin(theta[i]))# v eno smer ker gleda za poltrak
    r1 = np.add(r,d1) #pristejemo k rayorigin
    r2 = np.add(r,d2)

    for j in range(len(points2)):
        z1 =points2[j-1]
        z2 = points2[j]

        if lineRayIntersectionPoint(r,r1,z1,z2)==True or lineRayIntersectionPoint(r,r2,z1,z2)==True :
            num_hits+=1
            break
        

print("Stevilo presekov :",num_hits)  
verjetnost_1=num_hits/num_tries
print ("Verjetnost da seka obe hkrati eksperimentalno:",verjetnost_1)


vertices = hull.vertices.tolist() + [hull.vertices[0]]
perimeter = np.sum([euclidean(x, y) for x, y in zip(points[vertices], points[vertices][1:])]) #obseg objekta C

vertices1 = hull1.vertices.tolist() + [hull1.vertices[0]]
perimeter1 = np.sum([euclidean(x, y) for x, y in zip(points1[vertices1], points1[vertices1][1:])]) #obseg objekta C1

verjetnost_2 =perimeter1/perimeter
print("Razmerje med perimetrov konveksnih objektov: ",verjetnost_2)

primerjava= abs(verjetnost_2-verjetnost_1)
print("Primerjava :",primerjava)
print('_'*25)



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
        
      
        for i in range(len(b)):
          #  ax.axline(a[i],b[i], linewidth=1, color='k')
            r = b[i]
            d1 = ( 10*math.cos(theta[i]),10*math.sin(theta[i])) # v eno smer ker gleda za poltrak
            r1 = np.add(r,d1) #pristejemo k rayorigin
            
            d2 =((-10)*math.cos(theta[i]),(-10)*math.sin(theta[i]))
            r2 = np.add(r,d2) #pristejemo k rayorigin
            ax.axline(b[i],r1,color='k') 
            ax.plot(*r1.T, 'bo',markersize=3)
            ax.plot(*r2.T, 'yo',markersize=3)             
        
        ax.plot(*b.T, 'ro',markersize=3)
        
        
        plt.text(9,0, verjetnost_1 , fontsize=10, bbox=dict(facecolor='red', alpha=0.5)) #okno z ekspreimentalni rez
        
    ax.set_xticks(range(11)) # velikost osi 0-10
    ax.set_yticks(range(11))  
 
    
plt.show()








