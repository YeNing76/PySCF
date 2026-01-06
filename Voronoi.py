#!/usr/bin/env python
# coding: utf-8

import numpy as np


def becke_radial_grid(num_r, alpha, nu):
    # uniform grid in (0, 1)
    u = np.linspace(0.0, 1.0, num_r+2)[1:-1] # avoid u=0, and u=1 exactly

    # Becke stretching
    r = -alpha * np.log(1-u**nu)
    r = r[r>1e-3]
    return r 


# In[2]:


def lebedev_6():
    # 6-point Lebedev grid
    # 精确到 l = 1
    w = 4*np.pi/6
    pts = np.array([
        [ 1, 0, 0],
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0,-1, 0],
        [ 0, 0, 1],
        [ 0, 0,-1],
    ], dtype=float)
    weights = np.full(6, w)
    return pts, weights


# In[3]:


def lebedev_14():
    a = np.sqrt(1/3)
    pts = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
        [ a, a, a], [ a, a,-a],
        [ a,-a, a], [ a,-a,-a],
        [-a, a, a], [-a, a,-a],
        [-a,-a, a], [-a,-a,-a],
    ], dtype=float)

    w1 = 4*np.pi/21
    w2 = 4*np.pi/84
    weights = np.array([w1]*6 + [w2]*8)
    return pts, weights


# In[4]:


def lebedev_26():
    a = np.sqrt(1/3)
    b = np.sqrt(3/5)
    pts = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
        [ a, a, a], [ a, a,-a], [ a,-a, a], [ a,-a,-a],
        [-a, a, a], [-a, a,-a], [-a,-a, a], [-a,-a,-a],
        [ b, 0, 0], [-b, 0, 0],
        [ 0, b, 0], [ 0,-b, 0],
        [ 0, 0, b], [ 0, 0,-b],
        [ 0, a, b], [ 0, a,-b], [ 0,-a, b], [ 0,-a,-b],
        [ a, 0, b], [ a, 0,-b]
    ], dtype=float)

    w1 = 4*np.pi/30
    w2 = 4*np.pi/120
    w3 = 4*np.pi/60
    weights = np.array([w1]*6 + [w2]*8 + [w3]*12)
    return pts, weights


# In[5]:


import numpy as np

def lebedev_50():
    """
    Full Lebedev 50-point spherical grid.
    Returns:
        pts: (50,3) array of unit vectors
        w:   (50,) array of weights
    """

    pts = np.array([
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],

        [ 0.5773502691896257,  0.5773502691896257,  0.5773502691896257],
        [ 0.5773502691896257,  0.5773502691896257, -0.5773502691896257],
        [ 0.5773502691896257, -0.5773502691896257,  0.5773502691896257],
        [ 0.5773502691896257, -0.5773502691896257, -0.5773502691896257],
        [-0.5773502691896257,  0.5773502691896257,  0.5773502691896257],
        [-0.5773502691896257,  0.5773502691896257, -0.5773502691896257],
        [-0.5773502691896257, -0.5773502691896257,  0.5773502691896257],
        [-0.5773502691896257, -0.5773502691896257, -0.5773502691896257],

        [ 0.0,  0.5257311121191336,  0.85065080835204],
        [ 0.0,  0.5257311121191336, -0.85065080835204],
        [ 0.0, -0.5257311121191336,  0.85065080835204],
        [ 0.0, -0.5257311121191336, -0.85065080835204],

        [ 0.5257311121191336,  0.85065080835204,  0.0],
        [ 0.5257311121191336, -0.85065080835204,  0.0],
        [-0.5257311121191336,  0.85065080835204,  0.0],
        [-0.5257311121191336, -0.85065080835204,  0.0],

        [ 0.85065080835204,  0.0,  0.5257311121191336],
        [ 0.85065080835204,  0.0, -0.5257311121191336],
        [-0.85065080835204,  0.0,  0.5257311121191336],
        [-0.85065080835204,  0.0, -0.5257311121191336],

        [ 0.30901699437494745,  0.5,  0.8090169943749475],
        [ 0.30901699437494745,  0.5, -0.8090169943749475],
        [ 0.30901699437494745, -0.5,  0.8090169943749475],
        [ 0.30901699437494745, -0.5, -0.8090169943749475],
        [-0.30901699437494745,  0.5,  0.8090169943749475],
        [-0.30901699437494745,  0.5, -0.8090169943749475],
        [-0.30901699437494745, -0.5,  0.8090169943749475],
        [-0.30901699437494745, -0.5, -0.8090169943749475],

        [ 0.5,  0.8090169943749475,  0.30901699437494745],
        [ 0.5,  0.8090169943749475, -0.30901699437494745],
        [ 0.5, -0.8090169943749475,  0.30901699437494745],
        [ 0.5, -0.8090169943749475, -0.30901699437494745],
        [-0.5,  0.8090169943749475,  0.30901699437494745],
        [-0.5,  0.8090169943749475, -0.30901699437494745],
        [-0.5, -0.8090169943749475,  0.30901699437494745],
        [-0.5, -0.8090169943749475, -0.30901699437494745],

        [ 0.8090169943749475,  0.30901699437494745,  0.5],
        [ 0.8090169943749475,  0.30901699437494745, -0.5],
        [ 0.8090169943749475, -0.30901699437494745,  0.5],
        [ 0.8090169943749475, -0.30901699437494745, -0.5],
        [-0.8090169943749475,  0.30901699437494745,  0.5],
        [-0.8090169943749475,  0.30901699437494745, -0.5],
        [-0.8090169943749475, -0.30901699437494745,  0.5],
        [-0.8090169943749475, -0.30901699437494745, -0.5],
    ])

    # Weight（50 in total）
    w = np.array([
        0.126984126984127, 0.126984126984127,
        0.126984126984127, 0.126984126984127,
        0.126984126984127, 0.126984126984127,

        0.047619047619048, 0.047619047619048,
        0.047619047619048, 0.047619047619048,
        0.047619047619048, 0.047619047619048,
        0.047619047619048, 0.047619047619048,

        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
        0.075, 0.075, 0.075, 0.075,
    ])

    return pts, w


# In[6]:


import numpy as np

def lebedev_74():
    """
    Full Lebedev 74-point spherical grid.
    Returns:
        pts: (74,3) array of unit vectors
        w:   (74,) array of weights
    """

    a = 0.45970084338098305
    b = 0.6285393610547089
    c = 0.322185354626569
    d = 0.8360955967490217

    pts = np.array([
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],

        [ a,  a,  a], [ a,  a,-a], [ a,-a,  a], [ a,-a,-a],
        [-a,  a,  a], [-a,  a,-a], [-a,-a,  a], [-a,-a,-a],

        [ b,  0,  c], [ b,  0,-c], [-b,  0,  c], [-b,  0,-c],
        [ 0,  b,  c], [ 0,  b,-c], [ 0,-b,  c], [ 0,-b,-c],
        [ c,  b,  0], [ c,-b,  0], [-c,  b,  0], [-c,-b,  0],

        [ d,  d,  0], [ d,-d,  0], [-d,  d,  0], [-d,-d,  0],
        [ d,  0,  d], [ d,  0,-d], [-d,  0,  d], [-d,  0,-d],
        [ 0,  d,  d], [ 0,  d,-d], [ 0,-d,  d], [ 0,-d,-d],

        [ 0.0,  a,  b], [ 0.0,  a,-b], [ 0.0,-a,  b], [ 0.0,-a,-b],
        [ a,  0.0,  b], [ a,  0.0,-b], [-a,  0.0,  b], [-a,  0.0,-b],
        [ b,  a,  0.0], [ b,-a,  0.0], [-b,  a,  0.0], [-b,-a,  0.0],

        [ c,  c,  d], [ c,  c,-d], [ c,-c,  d], [ c,-c,-d],
        [-c,  c,  d], [-c,  c,-d], [-c,-c,  d], [-c,-c,-d],

        [ d,  c,  c], [ d,  c,-c], [ d,-c,  c], [ d,-c,-c],
        [-d,  c,  c], [-d,  c,-c], [-d,-c,  c], [-d,-c,-c],

        [ c,  d,  c], [ c,  d,-c], [ c,-d,  c], [ c,-d,-c],
        [-c,  d,  c], [-c,  d,-c], [-c,-d,  c], [-c,-d,-c],
    ])

    w1 = 0.047619047619047616
    w2 = 0.0380952380952381
    w3 = 0.03214285714285714
    w4 = 0.02857142857142857

    w = np.array(
        [w1]*6 +
        [w2]*8 +
        [w3]*12 +
        [w4]*48
    )

    return pts, w


# In[7]:


def lebedev_86():
    """
    Full Lebedev 86-point spherical grid.
    Returns:
        pts: (86,3) array of unit vectors
        w:   (86,) array of weights
    """

    a = 0.2666354015167047
    b = 0.681507726536546
    c = 0.4174961227965453
    d = 0.872473431981953

    pts = np.array([
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],

        [ a,  a,  a], [ a,  a,-a], [ a,-a,  a], [ a,-a,-a],
        [-a,  a,  a], [-a,  a,-a], [-a,-a,  a], [-a,-a,-a],

        [ b,  0,  c], [ b,  0,-c], [-b,  0,  c], [-b,  0,-c],
        [ 0,  b,  c], [ 0,  b,-c], [ 0,-b,  c], [ 0,-b,-c],
        [ c,  b,  0], [ c,-b,  0], [-c,  b,  0], [-c,-b,  0],

        [ d,  d,  0], [ d,-d,  0], [-d,  d,  0], [-d,-d,  0],
        [ d,  0,  d], [ d,  0,-d], [-d,  0,  d], [-d,  0,-d],
        [ 0,  d,  d], [ 0,  d,-d], [ 0,-d,  d], [ 0,-d,-d],

        [ 0.0,  a,  b], [ 0.0,  a,-b], [ 0.0,-a,  b], [ 0.0,-a,-b],
        [ a,  0.0,  b], [ a,  0.0,-b], [-a,  0.0,  b], [-a,  0.0,-b],
        [ b,  a,  0.0], [ b,-a,  0.0], [-b,  a,  0.0], [-b,-a,  0.0],

        [ c,  c,  d], [ c,  c,-d], [ c,-c,  d], [ c,-c,-d],
        [-c,  c,  d], [-c,  c,-d], [-c,-c,  d], [-c,-c,-d],

        [ d,  c,  c], [ d,  c,-c], [ d,-c,  c], [ d,-c,-c],
        [-d,  c,  c], [-d,  c,-c], [-d,-c,  c], [-d,-c,-c],

        [ c,  d,  c], [ c,  d,-c], [ c,-d,  c], [ c,-d,-c],
        [-c,  d,  c], [-c,  d,-c], [-c,-d,  c], [-c,-d,-c],
    ])

    w1 = 0.0380952380952381
    w2 = 0.03214285714285714
    w3 = 0.02857142857142857
    w4 = 0.025396825396825397

    w = np.array(
        [w1]*6 +
        [w2]*8 +
        [w3]*12 +
        [w4]*60
    )

    return pts, w


# In[8]:




def lebedev_grid(order):
    """
    Return Lebedev directions (unit vectors) and weights for a given order.
    order: one of {6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350}
    """
    if order == 6:
        return lebedev_6()
    elif order == 14:
        return lebedev_14()
    elif order == 26:
        return lebedev_26()
    elif order == 50:
        return lebedev_50()
    elif order == 74:
        return lebedev_74()
    elif order == 86:
        return lebedev_86()
    else:
        raise ValueError(f"Lebedev order {order} not implemented.")

# 3D adaptive grid for hydrogen using Lebedev 2025 12 21
from scipy.spatial import Voronoi
import numpy as np

def hydrogen_adaptive_grid_lebedev(num_r, order):
    # For the time being, we test the radial part via exponential stretch like r^1.5
    #r = np.linspace(0.1, r_max, num_r)**1.5
    alpha=1.0
    nu=2.0
    r = becke_radial_grid(num_r, alpha, nu)

    r = r[r > 1e-3]
    
    leb_pts, leb_w = lebedev_grid(order)
    points = []
    for ri in r:
        for n in leb_pts:
            points.append(ri * n)

    points = np.array(points)
    #vor = Voronoi(points, qhull_options='Qbb Qc Qx QJ')
    vor = Voronoi(points, qhull_options='QJ')
    
    neighbors = {i: set() for i in range(len(points))}            
    for p, q in vor.ridge_points:
        neighbors[p].add(q)
        neighbors[q].add(p)
    for m in range(len(points)):
        for n in neighbors[m]:
            if m not in neighbors[n]:
                print("Asym neighbor:", m, n)
    return np.array(points), neighbors, vor


# In[10]:

points, neighbors, vor = hydrogen_adaptive_grid_lebedev(num_r=60, order = 74)


# In[11]:


# calculate the geometrical properties of facet
def facet_geometry(vor, ridge_vertices):
    verts = vor.vertices[ridge_vertices]

    # normal
    v1 = verts[1] - verts[0]
    v2 = verts[2] - verts[0]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # centroid
    centroid = verts.mean(axis = 0)

    # area (shoelace)

    # Project polygon onto the plane where hte normal has the smallest component 
    # (this avoids degeneracy)
    ax = np.argmin(np.abs(normal))
    proj = np.delete(verts, ax, axis=1)

    # Compute 2D polygon area using shoelace formula
    x = proj[:, 0]
    y = proj[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -np.dot(y, np.roll(x, -1)))
    
    return normal, centroid, area


# In[12]:


# Calculate the facet areas
def facet_area(vor, ridge_vertices):
    if -1 in ridge_vertices:
        return np.inf # facet in infinity

    verts = vor.vertices[ridge_vertices]
    
    normal, _ = facet_geometry(vor, ridge_vertices)

    # Project polygon onto the plane where hte normal has the smallest component 
    # (this avoids degeneracy)
    ax = np.argmin(np.abs(normal))
    proj = np.delete(verts, ax, axis=1)

    # Compute 2D polygon area using shoelace formula
    x = proj[:, 0]
    y = proj[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -np.dot(y, np.roll(x, -1)))
    return area


# In[13]:


# Calculate Voronoi cell volumes
def cell_volume(vor, points, neighbors):
    sigma = {} # dictionary of dictionaries, to hold the information of area
    normals = {} # to hold the imformation of normal
    centroids = {} # to hold the information of centroid of a facet
    volumes = [0.0] * len(points)

    
    # This loop is to set up the dictionaries of sigma, normals and centroids
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        # skip infinite facets
        if -1 in rv:
            continue
            
        normal, centroid, area = facet_geometry(vor, rv)
        normals.setdefault(p, {})[q] = normal
        normals.setdefault(q, {})[p] = normal
        centroids.setdefault(p, {})[q] = centroid
        centroids.setdefault(q, {})[p] = centroid
        sigma.setdefault(p, {})[q] = area
        sigma.setdefault(q, {})[p] = area

    for m in range(len(points)):
        r_m = points[m]
        volume_m = 0.0

        for n in neighbors[m]:

            # skip missing facets (infinite or degenerate)
            if n not in sigma.get(m, {}):
                continue
                
            area = sigma[m][n]
            normal = normals[m][n]
            centroid = centroids[m][n]

            height = abs(np.dot(normal, centroid - r_m))
            volume_m += area * height / 3.0
            
        volumes[m] = volume_m
 
    return sigma, volumes
    


# In[14]:


# Plot the 3d hydrogen points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#points = hydrogen_adaptive_grid()

r = np.linalg.norm(points, axis = 1)

fig = plt.figure(figsize= (6, 6))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(points[:,0], points[:,2], points[:, 2], c=r, cmap='viridis', s=12)
fig.colorbar(p, ax=ax, label='radius')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Hydrogen Adaptive Grid (colored by radius)')

plt.show()


# In[15]:


# Laplacian matrix from (15)
def lapl_matr(vor, points, neighbors):
    sigma, volumes = cell_volume(vor, points, neighbors)
    length = len(points)

    # initialize Laplacian matrix
    lapl = [[0.0]*length for _ in range(length)]

    for m in range(length):
        r_m = points[m]

        # diagonal term
        diag_sum = 0.0
        for k in neighbors[m]:
            if k not in sigma[m]:
                continue
            dist = np.linalg.norm(points[m] - points[k])
            diag_sum += sigma[m][k] / dist

        lapl[m][m] = -diag_sum / volumes[m]
        #lapl[m][m] = -diag_sum / np.sqrt(volumes[m]*volumes[n])

       # off-diagonal terms
        for n in neighbors[m]:
            if n not in sigma[m]:
                continue
            dist = np.linalg.norm(points[m] - points[n])
            lapl[m][n] = sigma[m][n] / (np.sqrt(volumes[m]*volumes[n]) * dist)
            # Expression above is from the Eqn. (21)

    return lapl


# In[16]:


# Coulombic electron-nuclei attraction
# For hydrogen atom, we set the nuclear at the origin
# Therefore, \vec{R}_{\alpha}=\vec{0} and Z_{\alpha}=1
def coulombic_potential(points):
    length = len(points)
    ele_nu = [0.0] * length
    ele_ele = [[0.0]*length for _ in range(length)]

    Z_alpha = 1.0
    R_alpha = np.array([0.0, 0.0, 0.0])

    # electron–nucleus attraction
    for m in range(length):
        dist_nucl = np.linalg.norm(points[m] - R_alpha)
        if dist_nucl == 0:
            ele_nu[m] = np.inf
        else:
            ele_nu[m] = -Z_alpha / dist_nucl   # negative sign is important

    # electron–electron repulsion
    for m in range(length):
        for p in range(m, length):
            if m == p:
                ele_ele[m][p] = 0.0
            else:
                dist = np.linalg.norm(points[m] - points[p])
                ele_ele[m][p] = 1.0 / dist
                ele_ele[p][m] = ele_ele[m][p]  # symmetry

    return ele_nu, ele_ele


# In[23]:


# For the time being, let me establish Hamiltonian via (14) for hydrogen 2025 12 16
def hamilton_14(vor, points, neighbors):
    length = len(points)
    ham = [[0.0]*length for _ in range(length)]
    lapl = lapl_matr(vor, points, neighbors) # Calculate the Laplacian
    ele_nu, ele_ele = coulombic_potential(points)
    for m in range(length):
        for n in range(length):
            ham[m][n] = -0.5*lapl[m][n]

    for m in range(length):
        ham[m][m] += ele_nu[m]
    return ham


# In[24]:


import numpy as np
ham = hamilton_14(vor, points, neighbors)
H = np.array(ham)
vals, vecs = np.linalg.eigh(H)
print(vals[0])
