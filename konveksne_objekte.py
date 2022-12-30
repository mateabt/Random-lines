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

points1 = npi.difference(points,points[hull.vertices]) #tocke brez tiste na ovojnici C1
hull1= ConvexHull(points1) 

#Bounding box
bbox = [hull.min_bound, hull.max_bound]

num_hits = 0 # gre skozi obe hkrati
num_tries = 4 #koliko krat NE gre skozi obe (stevilo premic)
num_miss =0

trikotniki= len(list(combinations(points1, 3) )) # koliko krat bomo preverili razlicne kombinacije na trikotnike ce seka
# z koordinata bomo dali 1

def magnitude(vector):
   return np.sqrt(np.dot(np.array(vector),np.array(vector)))

def norm(vector):
   return np.array(vector)/magnitude(np.array(vector))


# ne dela pravilno
def lineRayIntersectionPoint(rayOrigin, rayDirection, point1, point2):
    rayOrigin = np.array(rayOrigin, dtype=np.float)
    rayDirection = np.array(norm(rayDirection), dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)

    v1 = rayOrigin - point1
    v2 = point2 - point1
    v3 = np.array([-rayDirection[1], rayDirection[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        return [rayOrigin + t1 * rayDirection]
    return []




n = num_tries # stevilo premic
b= np.empty((n, 2))
for i in range(n):
    #nakljucna tocka v C skozi katero grejo
    b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
    #Preverimo ce je res v c           
    while Path(hull.points[hull.vertices] ).contains_point(b[i]) == False:
        b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])


theta=[]
for i in range(len(b)):
               theta=np.append(theta,np.random.uniform(0,2*np.pi))

for i in range(len(b)):
    r = b[i]
    d = ((math.cos(theta[i]),math.sin(theta[i])))
    for j in range(len(points1)):
        z1 =points1[j-1]
        z2 = points1[j]
        if len(lineRayIntersectionPoint(r,d,z1,z2)) == 1:
            num_hits+=1
            break
        break
    
print(num_hits)   
#def line_triangle_intersection(
#    vertices: np.ndarray,
#    ray_origin: np.ndarray,
#    ray_direction: np.ndarray,
#    culling: bool = False,
#    epsilon: float = 1e-6,
#) -> Union[bool, Tuple[float, float, float]]:
#    """
#    kako naj definiram naklpn premice?
#   
#    """
#    vertex_0 = vertices[0]
#    vertex_1 = vertices[1]
#    vertex_2 = vertices[2]
#
#    edge_1 = vertex_1 - vertex_0
#    edge_2 = vertex_2 - vertex_0
#
#    p_vec = np.cross(ray_direction, edge_2)
#
#    determinant = np.dot(p_vec, edge_1)
#
#    if culling:
#        if determinant < epsilon:
#            return False
#
#        t_vec = ray_origin - vertex_0
#        u_ = np.dot(p_vec, t_vec)
#        if u_ < 0.0 or u_ > determinant:
#            return False
#
#        q_vec = np.cross(t_vec, edge_1)
#        v_ = np.dot(q_vec, ray_direction)
#        if v_ < 0.0 or (u_ + v_) > determinant:
#            return False
#
#        inv_determinant = 1.0 / determinant
#        t = np.dot(q_vec, edge_2) * inv_determinant
#        u = u_ * inv_determinant
#        v = v_ * inv_determinant
#
#        return t, u, v
#
#    else:
#        if np.abs(determinant) < epsilon:
#            return False
#
#        inv_determinant = 1.0 / determinant
#
#        t_vec = ray_origin - vertex_0
#        u = np.dot(p_vec, t_vec) * inv_determinant
#        if u < 0.0 or u > 1.0:
#            return False
#
#        q_vec = np.cross(t_vec, edge_1)
#        v = np.dot(q_vec, ray_direction) * inv_determinant
#        if v < 0.0 or (u + v) > 1.0:
#            return False
#
#        t = np.dot(q_vec, edge_2) * inv_determinant
#        if t < epsilon:
#            return False
#        return t, u, v
##Morda drugo tocko ce najdemo
## k=(y2-y1)/(x2-x1)
## k= dy/dx
#
#
##### ze bodo ponavljale XXXXX
#for i in range(0, num_tries): #za vsaka premica
#    for j in range(0, trikotniki):# vsak trikotnik
#        vertices=(list(combinations(points1, 3)))[j]
#        xyz = np.append(vertices,1)
#        nov_bi=np.append(b[i],0)
#        t = line_triangle_intersection(vertices,nov_bi,np.array[0,0,slope[i]])
#        if t != False:
#          num_hits += 1
#        else:
#          num_miss += 1
#          
#Verjetnost=num_hits/num_tries
#print(Verjetnost)


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

