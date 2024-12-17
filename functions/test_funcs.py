from __future__ import print_function, absolute_import, division

__all__ = ['Particles', 'inv_distance', 'log_distance']

import numpy as np
from time import time
from numba import jit
import math
import cmath
from scipy import integrate as intg


def log_dist_int(x,y):
    return -1 / (2 * np.pi) * np.log(np.sqrt(x ** 2 + y ** 2))
def log_dist_2d(xd,yd):
    return -1 / (2 * np.pi) * np.log(np.sqrt((xd[0] - yd[0]) ** 2 + (xd[1] - yd[1]) ** 2))
def log_distance(data1, list1, data2, list2):
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    vertex1 = data1.vertex
    vertex2 = data2.vertex
    n = list1.size
    m = list2.size
    N = data1.vertex.shape[1]
    for i in range(n):
        for j in range(m):
            if (vertex1[:,list1[i]] == vertex2[:,list2[j]]).all():
                ans[i, j] = intg.dblquad(log_dist_int,0,1/(2*np.sqrt(N)),lambda x: 0, lambda x: 1/(2*np.sqrt(N)))[0]*4
            else:
                ans[i, j] = log_dist_2d(vertex1[:,list1[i]],vertex2[:,list2[j]])/N
    return ans

###############################################################################
###                   interactions for Particles                            ###
###############################################################################

def inv_distance(data1, list1, data2, list2):
    """
    Returns 1/r for each pair of particles from two sets.

    Function 1/r is used as interaction between two particles.

    Parameters
    ----------
    data1 : Python object
        Destination of interactions
    list1 : array
        Indices of particles from `data1` to compute interactions
    data2 : Python object
        Source of interactions
    list2 : array
        Indices of particles from `data1` to compute interactions

    Returns
    -------
    numpy.ndarray(ndim=2)
        Array of interactions of corresponding particles.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return inv_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, data1.count, data1.d_c, ans)

@jit(nopython=True)
def inv_distance_numba(ndim, vertex1, list1, vertex2, list2, count, d_c, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]]-vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                if d_c is None:
                    ans[i, j] = 0
                else:
                    ans[i, j] = d_c
                # ans[i, j] = 600
                # ans[i, j] =  np.sqrt(count)*8
            else:
                ans[i, j] = 1./math.sqrt(tmp_l)
                # ans[i, j] = 1./tmp_l
    return ans

def log_distance_h2t(data1, list1, data2, list2):
    """
    Returns -log(r) for each pair of particles from two sets.

    Function -log(r) is used as interaction between two particles.

    Parameters
    ----------
    data1 : Python object
        Destination of interactions
    list1 : array
        Indices of particles from `data1` to compute interactions
    data2 : Python object
        Source of interactions
    list2 : array
        Indices of particles from `data1` to compute interactions

    Returns
    -------
    numpy.ndarray(ndim=2)
        Array of interactions of corresponding particles.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return log_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, ans)

@jit(nopython=True)
def log_distance_numba(ndim, vertex1, list1, vertex2, list2, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]] - vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                # ans[i, j] = -1 * math.log(np.sqrt(tmp_l))
                ans[i, j] = -1 * math.log(tmp_l)
            if list1[i] == list2[j]:
                ans[i, j] = 4
    return ans

def exp_distance_h2t(data1, list1, data2, list2):
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return exp_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, data1.d_c, data1.sigma, ans)

def gauss_distance_h2t(data1, list1, data2, list2):
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return gauss_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, data1.d_c,data1.sigma, ans)

def exp_fun(data1, list1, data2, list2):
    sigma = 0.1
    n = list1.size
    m = list2.size
    vertex1 = data1.vertex
    vertex2 = data2.vertex
    N = data1.vertex.shape[1]
    ans = np.ndarray((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            d2 = 0
            for i_dim in range(data1.ndim):
                d2 += (vertex1[i_dim,list1[i]] - vertex2[i_dim,list2[j]])**2
            d2 = np.sqrt(d2)
            if (vertex1[:,list1[i]] == vertex2[:,list2[j]]).all():
                ans[i, j] = np.exp(-d2/sigma) * diag_coef
            else:
                ans[i, j] = np.exp(-d2/sigma)
    return ans

@jit(nopython=True)
def exp_distance_numba(ndim, vertex1, list1, vertex2, list2, d_c, sigma, ans):
    n = list1.size
    m = list2.size
    # diag_coef = d_c
    # sigma = 0.1
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]] - vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            # if tmp_l <= 0:
            #     ans[i, j] = 0
            # else:
            r = math.sqrt(tmp_l)
            if list1[i] == list2[j]:
                if d_c is None:
                    ans[i, j] = np.exp(-r/sigma)
                else:
                    ans[i, j] = np.exp(-r/sigma) * d_c
            else:
                ans[i,j] = np.exp(-r/sigma)

    return ans

@jit(nopython=True)
def gauss_distance_numba(ndim, vertex1, list1, vertex2, list2, d_c, sigma, ans):
    n = list1.size
    m = list2.size
    # diag_coef = 4
    # sigma = 0.01
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]] - vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            # if tmp_l <= 0:
            #     ans[i, j] = 0
            # else:
            # r = math.sqrt(tmp_l)
            r = tmp_l
            if list1[i] == list2[j]:
                if d_c is None:
                    ans[i, j] = np.exp(-r/sigma)
                else:
                    ans[i, j] = np.exp(-r/sigma) * d_c
            else:
                ans[i,j] = np.exp(-r/sigma)

    return ans

@jit(nopython=True)
def exp_distance_numba_old(ndim, vertex1, list1, vertex2, list2, kz, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]] - vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                r = math.sqrt(tmp_l)
                ans[i, j] = cmath.exp(1j * kz * r)/ r
            if list1[i] == list2[j]:
                ans[i, j] = 6. + 1j*0
    return ans


def double_layer(data1, list1, data2, list2):
    ans = np.ndarray((list1.size, list2.size), dtype=np.cdouble)
    return double_layer_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, data1.k, data1.norms, ans)

@jit(nopython=True)
def double_layer_numba(ndim, vertex1, list1, vertex2, list2, kz, norms, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            tetha = 0.0
            len_norm = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]] - vertex2[k, list2[j]]
                tmp_l += tmp_v * tmp_v
                tetha += tmp_v * norms[k,list1[i]]
                len_norm += norms[k,list1[i]] * norms[k,list1[i]]
#             print (tetha)
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                r = math.sqrt(tmp_l)
                len_norm = math.sqrt(len_norm)
                tetha = tetha / (r * len_norm)
                ans[i, j] = (cmath.exp(1j * kz * r)/ r) * (1j * kz - 1/r)* math.cos(tetha)
            if list1[i] == list2[j]:
                ans[i, j] = 6 + 1j*0
    return ans

@jit(nopython=True)
def comp_sph_numba(ndim, vertex1, list1, vertex2, list2, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]]-vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                ans[i, j] = 1/(4 * np.pi * math.sqrt(tmp_l))
            if list1[i] == list2[j]:
                ans[i, j] = 150
    return ans

def comp_sph(data1, list1, data2, list2):
    ans = np.ndarray((list1.size, list2.size))
    return comp_sph_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, ans)
