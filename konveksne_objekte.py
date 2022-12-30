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

trikotniki= len(list(combinations(points1, 3) )) # koliko krat bomo preverili razlicne kombinacije na trikotnike ce seka

# z koordinata bomo dali 1


n = num_tries # stevilo premic
b= np.empty((n, 2))
for i in range(n):
    #nakljucna tocka v C skozi katero grejo
    b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
    #Preverimo ce je res v c           
    while Path(hull.points[hull.vertices] ).contains_point(b[i]) == False:
        b[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])




def ray_triangle_intersection(
    vertices: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    culling: bool = False,
    epsilon: float = 1e-6,
) -> Union[bool, Tuple[float, float, float]]:
    """
    Examples
    --------
    >>> vertices = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    >>> ray_origin = np.array([1.0, 1.0, 1.0])
    >>> ray_direction = np.array([0.0, 0.0, -1.0])
    >>> intersection = ray_triangle_intersection(vertices, ray_origin, ray_direction)
    (1.0, 0.1, 0.1)
    """
    vertex_0 = vertices[0]
    vertex_1 = vertices[1]
    vertex_2 = vertices[2]

    edge_1 = vertex_1 - vertex_0
    edge_2 = vertex_2 - vertex_0

    p_vec = np.cross(ray_direction, edge_2)

    determinant = np.dot(p_vec, edge_1)

    if culling:
        if determinant < epsilon:
            return False

        t_vec = ray_origin - vertex_0
        u_ = np.dot(p_vec, t_vec)
        if u_ < 0.0 or u_ > determinant:
            return False

        q_vec = np.cross(t_vec, edge_1)
        v_ = np.dot(q_vec, ray_direction)
        if v_ < 0.0 or (u_ + v_) > determinant:
            return False

        inv_determinant = 1.0 / determinant
        t = np.dot(q_vec, edge_2) * inv_determinant
        u = u_ * inv_determinant
        v = v_ * inv_determinant

        return t, u, v

    else:
        if np.abs(determinant) < epsilon:
            return False

        inv_determinant = 1.0 / determinant

        t_vec = ray_origin - vertex_0
        u = np.dot(p_vec, t_vec) * inv_determinant
        if u < 0.0 or u > 1.0:
            return False

        q_vec = np.cross(t_vec, edge_1)
        v = np.dot(q_vec, ray_direction) * inv_determinant
        if v < 0.0 or (u + v) > 1.0:
            return False

        t = np.dot(q_vec, edge_2) * inv_determinant
        if t < epsilon:
            return False
        return t, u, v

class Vec3: #vektor v R3
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def sub(self, v):
        return Vec3(self.x - v.x,
                    self.y - v.y,
                    self.z - v.z)

    def dot(self, v): #skalaren
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v): #vektorski
        return Vec3(self.y * v.z - self.z * v.y,
                    self.z * v.x - self.x * v.z,
                    self.x * v.y - self.y * v.x)

    def length(self): #dolzina
        return math.sqrt(self.x * self.x +
                         self.y * self.y +
                         self.z * self.z)
        
    def normalize(self): #normalizacija
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l)


num_miss =0
slope=[]
for i in range(len(b)):
               slope=np.append(slope,np.random.uniform((-np.pi)/2,(np.pi)/2))


for i in range(0, num_tries): #za vsaka premica
    for j in range(0, trikotniki):# vsak trikotnik
        vertices=(list(combinations(points1, 3)))[j]
        xyz = np.append(vertices,1)
        nov_bi=np.append(b[i],0)
        t = ray_triangle_intersection(vertices,nov_bi,np.array[0,0,slope[i]])
        if t >= 0:
          num_hits += 1
        else:
          num_miss += 1
#vertices = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
 #   >>> ray_origin = np.array([1.0, 1.0, 1.0])
 #   >>> ray_direction = np.array([0.0, 0.0, -1.0])
 #   >>> intersection = ray_triangle_intersection(vertices, ray_origin, ray_direction)
 #   (1.0, 0.1, 0.1)



#def ray_triangle_intersect(r, v0, v1, v2): #presek premice z trikotnik
#    v0v1 = v1.sub(v0)
#    v0v2 = v2.sub(v0)
#    pvec = r.direction.cross(v0v2)
#
#    det = v0v1.dot(pvec)
#
#    if det < 0.000001:
#        return float('-inf')
#
#    invDet = 1.0 / det
#    tvec = r.orig.sub(v0)
#    u = tvec.dot(pvec) * invDet
#
#    if u < 0 or u > 1:
#        return float('-inf')
#
#    qvec = tvec.cross(v0v1)
#    v = r.direction.dot(qvec) * invDet
#
#    if v < 0 or u + v > 1:
#        return float('-inf')
#
#    return v0v2.dot(qvec) * invDet
#
#
#
# 
#
#
#
#
#for i in range(0, num_tries): #za vsaka premica
#    for j in range(0, trikotniki):   # vsak trikotnik
#        t = ray_triangle_intersect(r, vertices[j*3 + 0],
#                                      vertices[j*3 + 1],
#                                      vertices[j*3 + 2])
#        if t >= 0:
#          num_hits += 1
#        else:
#          num_miss += 1
#
#
#    
#
#
#
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
            ax.axline(b[i],slope[i],color='k')
                      
        # a modre b rdece
        #ax.plot(*a.T, 'bo',markersize=3)
        ax.plot(*b.T, 'ro',markersize=3)
        
        
        
    ax.set_xticks(range(11)) # velikost osi 0-10
    ax.set_yticks(range(11))  
plt.show()

