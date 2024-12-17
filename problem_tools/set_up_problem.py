import numpy as np
import sys
sys.path.insert(0,'..')
from copy import deepcopy as dc
from collections import defaultdict
from itertools import product
from time import time
from scipy.linalg import cho_factor, cho_solve, cholesky, lu
from scipy.linalg.blas import dgemm
from problem_tools.geometry_tools import Data, Tree
from problem_tools.problem import Problem
from functions import test_funcs
from scipy.linalg.interpolative import interp_decomp
import scipy.sparse as sps

def build_cube_problem(func, n=15, ndim=2, block_size=28, verbose=1,point_based_tree = True, close_r='1box',num_child_tree = 'hyper', random_points=0, zk=None,diag_coef=None,sigma=0.1):
    count = n**ndim
    if random_points:
        position = np.random.rand(ndim,n**ndim)
    else:
        if ndim == 1:
            x0 = np.arange(1, n+1) / n
            position = x0.reshape(1, n**ndim)
        elif ndim == 2:
            x0, x1 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n)
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim)))
        elif ndim == 3:
            x0, x1, x2 = np.meshgrid(np.arange(1,n+1)/(n),np.arange(1,n+1)/n, np.arange(1,n+1)/(n))
            position = np.vstack((x0.reshape(1,n**ndim),x1.reshape(1,n**ndim), x2.reshape(1,n**ndim)))

    data = Data(ndim, count, position, close_r=close_r)
    if func == test_funcs.exp_distance_h2t:
        data.k = zk
    data.d_c = diag_coef
    data.sigma = sigma
    tree = Tree(data, block_size, point_based_tree = point_based_tree, num_child_tree = num_child_tree)
    problem = Problem(func, tree, verbose%2)
    return problem
def build_problem(geom_type='cube',block_size=26, n=15, ndim = 2, func = test_funcs.log_distance, point_based_tree=0, close_r = 1., num_child_tree='hyper', random_points=1, file = None, eps = 0.51e-6, zk = 1.1 + 1j*0, alpha = 3.0, beta = 0, wtd_T=0,add_up_level_close=0,half_sym=0,csc_fun=0,q_fun=0,ifwrite=0, nu=10, order=10, diag_coef=None, sigma=0.1):
    iters = 2
    onfly = 1
    verbose = 0
    random_init = 2
    if point_based_tree and close_r == '1box':
        print("!!'1box' does not work with point_based_tree = 1, close_r chanjed to 1.")
        close_r = 1.
    if geom_type == 'cube':
        pr = build_cube_problem(func, n=n, ndim=ndim, block_size=block_size,
                                  verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,random_points = random_points,zk=zk, diag_coef=diag_coef,sigma=sigma)
    elif geom_type == 'sphere':
        pr = build_sphere(func, n=n, ndim=ndim, block_size=block_size,
                                   verbose=verbose, point_based_tree=point_based_tree,
                                  close_r=close_r,num_child_tree = num_child_tree,zk=zk)
    elif geom_type == 'from_file':
        if file is None:
            raise NameError(f"Geometry type '{geom_type}' should have nonempty file!")
        pr = build_problem_from_file(func, block_size=block_size, verbose=verbose,
                                     point_based_tree=point_based_tree, close_r=close_r,
                                     num_child_tree=num_child_tree, file=file,eps=eps, zk=zk, alpha=alpha,
                                     beta=beta,csc_fun=csc_fun,ifwrite=ifwrite)
    elif geom_type == 'wtorus':
        pr = build_problem_wtorus(func, block_size=block_size, verbose=verbose,
                                  point_based_tree=point_based_tree, close_r=close_r,
                                  num_child_tree=num_child_tree, eps=eps, zk=zk, alpha=alpha,
                                  beta=beta,csc_fun=csc_fun,ifwrite=ifwrite, nu=nu, order=order)
    else:
        raise NameError (f"Geometry type '{geom_type}' is not supported. Try 'qube/sphere/from_file/wtorus'")
    pr.add_up_level_close = add_up_level_close
    if add_up_level_close:
        print('Warning! up-level close is not well-tested!')
    n_parants = 1
    for i in range(1, len(pr.tree.level)-1):
        n_nodes_lvl = pr.tree.level[i+1] - pr.tree.level[i]
        if n_nodes_lvl != n_parants * pr.tree.nchild:
            pr.tree.higest_leaf_lvl = i-1
            break
        n_parants = n_nodes_lvl
    pr.csc_fun = csc_fun
    pr.wtd_T = wtd_T
    pr.half_sym = half_sym
    pr.q_fun = q_fun
    pr.eps = eps
    tree = pr.tree
    level_count = len(tree.level) - 2
    for i in range(level_count-1, -1, -1):
        job = [j for j in
                        range(tree.level[i], tree.level[i+1])]
        exist_no_trans_t = False
        exist_no_trans_f = False
        for ind in job:
            if pr.notransition[ind]:
                exist_no_trans_t = True
            else:
                exist_no_trans_f = True
        if exist_no_trans_t and exist_no_trans_f:
            print ('lvl', i, '+-')
            pr.tail_lvl = i
            for ind in job:
                pr.notransition[ind] = False
        elif exist_no_trans_t:
            pr.tail_lvl = i+1
            break
    return pr
