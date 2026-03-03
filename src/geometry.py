import numpy as np
from scipy.spatial import Voronoi
from collections import defaultdict

"""
The following function is to realize Eqn. (5) in the paper
"""

def becke_radial_grid(num_r, alpha, nu):
    # uniform grid in (0, 1)
    u = np.linspace(0.0, 1.0, num_r+2)[1:-1] # avoid u=0, and u=1 exactly

    # Becke stretching
    r = -alpha * np.log(1-u**nu)
    #r = r[r>1e-3]
    return r 

"""
The following several functions are about Lebedev grids of different orders
"""
# Lebedev 6
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

# lebedev 14
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

# Lebedev 26
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

# Lebedev 50
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

    # 权重（全部 50 个）
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

# Lebedev 74
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

# Lebedev 86
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

# Select different Lebedev grids
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

def hydrogen_adaptive_grid_lebedev(num_r, order):
    # For the time being, we test the radial part via exponential stretch like r^1.5
    #r = np.linspace(0.1, r_max, num_r)**1.5
    alpha=1.0
    nu=2.0
    r = becke_radial_grid(num_r, alpha, nu)

    # r = r[r > 1e-3]
    
    leb_pts, leb_w = lebedev_grid(order)
    points = []
    for ri in r:
        for n in leb_pts:
            points.append( ri * n)

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

# calculate the geometrical properties of facet
def facet_geometry(vor, ridge_vertices):
    verts = vor.vertices[ridge_vertices]
    # compute centroid
    c = verts.mean(axis=0)

    # compute angles
    # angles = np.arctan2(verts[:,1] - c[1], verts[:,0] - c[0])

    # sort vertices by angle
    #order = np.argsort(angles)
   # verts = verts[order]

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
    c2 = proj.mean(axis=0)

    # compute angles
    angles = np.arctan2(proj[:,1] - c2[1], proj[:,0] - c2[0])

    # sort vertices by angle
    order = np.argsort(angles)
    proj = proj[order]

    # Compute 2D polygon area using shoelace formula
    x = proj[:, 0]
    y = proj[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -np.dot(y, np.roll(x, -1)))
    
    return normal, centroid, area

def cell_volume(points, neighbors, sigma):
    volumes = [0.0] * len(points)

    for m in range(len(points)):
        for n in neighbors[m]:
            if n not in sigma.get(m, {}):
                continue

            area = sigma[m][n] 
            dist = np.linalg.norm(points[m] - points[n])

            volumes[m] += area * dist / 6.0

    return volumes           


# 几何预处理接口（Geometry Preprocessing）2026 02 20
def build_geometry(points, neighbors, vor):
    sigma = defaultdict(dict)

    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            continue
        normal, centroid, area = facet_geometry(vor, rv)
        sigma[p][q] = area
        sigma[q][p] = area

    #neighbors = build_neighbors(vor)
    volumes = cell_volume(points, neighbors, sigma)

    return sigma, volumes

