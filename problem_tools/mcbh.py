"""
MCBH method of approximation of problem matrix by H2-matrix.
"""

from __future__ import print_function, absolute_import, division
# import sys
# sys.path.append("../")
from time import time
import numpy as np
from .maxvol import maxvol_svd, maxvol_qr
from .h2matrix import H2matrix

__all__ = ['mcbh']

def mcbh(problem, tau, alpha=0., iters=1, onfly=False, verbose=False, random_init=0, mpi_comm=None):

    """
    Computes H2-approximation by MCBH method.

    Uses Multicharge Barnes-Hut method (MCBH) of approximation of
    discretized non-local operators by H2-matrices [MCBH]_.

    Parameters
    ----------
    problem : h2tools.problem.Problem
        Discretized problem with block cluster tree.
    tau : float
        Parameter of desired relative error (spectral tolerance for
        SVDs).
    alpha : float
        Regularizing parameter for `maxvolpy.maxvol.maxvol_svd`.
    iter : integer
        Number of iterations (depends on problem).
    onfly : bool
        Wheather or not to use "low" memory allocation type (useful
        if matrix elements are fast enough to compute, saves a lot
        of memory).
    verbose : boolean
        Whether or not to print additional information.
    random_init : integer
        Number of basis elements to intializate hierarchical basises
        randomly.
    mpi_comm : MPI communicator

    Returns
    -------
    h2tools.h2matrix.H2matrix
        H2-approximation.
    """
    time0 = time()
    # far field approximation timings [ function time,
    #     block function calls, elements computed, maxvol time]
    far_timings = [0., 0, 0, 0.]
    # close field timings [ function time, block function calls,
    #     elements computed]
    close_timings = [0., 0, 0]
    # Initialization of MCBH method with upward pass
    row_size = problem.tree.level[-1]
    row_prebasis = [np.ndarray(0, dtype=np.uint64) for i in range(row_size)]
    col_size = problem.tree.level[-1]
    col_prebasis = [np.ndarray(0, dtype=np.uint64) for i in range(col_size)]
    # Initialization of parent far-field zone with random rows/columns
    if random_init > 0:
        row_basis, col_basis = _upward_random(problem, random_init, mpi_comm)
        row_prebasis, col_prebasis = _downward(problem, row_basis, col_basis,\
                far_timings, verbose, mpi_comm)
    row_basis_size, row_basis, row_transfer, col_basis_size, col_basis,\
            col_transfer = _upward(problem, tau, alpha, row_prebasis,\
            col_prebasis, far_timings, verbose, mpi_comm)
    # Iterations of MCBH method
    for i in range(iters):
        row_prebasis, col_prebasis = _downward(problem, row_basis,\
                col_basis, far_timings, verbose, mpi_comm)
        row_basis_size, row_basis, row_transfer, col_basis_size, col_basis,\
                col_transfer = _upward(problem, tau, alpha, row_prebasis,\
                col_prebasis, far_timings, verbose, mpi_comm)
    # Initialize near-field interaction matrices (M2L in terms of FMM)
    #   and close-field interaction matrices
    row_interaction = None
    col_interaction = None
    row_close = None
    col_close = None
    return row_basis_size, col_basis_size, row_transfer,col_transfer, row_interaction, col_interaction, row_close, col_close, row_basis, col_basis, mpi_comm
    # Init instance of H2matrix with given representation
    # ans = H2matrix(problem, row_basis_size, col_basis_size, row_transfer,
    #         col_transfer, row_interaction, col_interaction, row_close,
    #         col_close, row_basis, col_basis, mpi_comm)
    # if not onfly:
    #     ans._matrix(far_timings)
    #     ans._close_matrix(close_timings)
    #     ans.mem_type = 'full'
    # totaltime = time()-time0
    # # Output information
    # if verbose and mpi_comm is not None:
    #     mpi_rank = mpi_comm.rank
    #     mpi_size = mpi_comm.size
    #     tmp_buffer = mpi_comm.gather((far_timings, close_timings,
    #         ans.nbytes(0, 0, 0, 1), ans.nbytes(1, 0, 0, 0),
    #         ans.nbytes(0, 1, 0, 0), ans.nbytes(0, 0, 1, 0,),
    #         ans.nbytes(), totaltime), root=0)
    #     if mpi_rank == 0:
    #         far_timings = [np.zeros(mpi_size, np.float)]\
    #                 +[np.zeros(mpi_size, np.int)]\
    #                 +[np.zeros(mpi_size, np.int)]\
    #                 +[np.zeros(mpi_size, np.float)]
    #         close_timings = [np.zeros(mpi_size, np.float)]\
    #                 +[np.zeros(mpi_size, np.int)]\
    #                 +[np.zeros(mpi_size, np.int)]
    #         mem = np.zeros((5, mpi_size), np.float)
    #         totaltime = np.zeros(mpi_size, np.float)
    #         for i in range(mpi_size):
    #             ii = tmp_buffer[i]
    #             far_timings[0][i] = ii[0][0]
    #             far_timings[1][i] = ii[0][1]
    #             far_timings[2][i] = ii[0][2]
    #             far_timings[3][i] = ii[0][3]
    #             close_timings[0][i] = ii[1][0]
    #             close_timings[1][i] = ii[1][1]
    #             close_timings[2][i] = ii[1][2]
    #             mem[0][i] = ii[2]
    #             mem[1][i] = ii[3]
    #             mem[2][i] = ii[4]
    #             mem[3][i] = ii[5]
    #             mem[4][i] = ii[6]
    #             totaltime[i] = ii[7]
    #         mem /= (2**20)
    #         old_printoptions = np.get_printoptions()
    #         np.set_printoptions(precision=2, suppress=1)
    #         print('Far-field interactions(MCBH method):')
    #         print('    Function calls: {}'.format(far_timings[1]))
    #         print('    Function values computed: {}'.format(far_timings[2]))
    #         print('    Function time, seconds: {}'.format(far_timings[0]))
    #         print('    Average time per function value, seconds: {}'.
    #                 format(far_timings[0]/far_timings[2]))
    #         print('    Maxvol time, seconds: {}'.format(far_timings[3]))
    #         print('Near-field interactions:')
    #         print('    Function calls: {}'.format(close_timings[1]))
    #         print('    Function values computed: {}'.format(close_timings[2]))
    #         print('    Function time, seconds: {}'.format(close_timings[0]))
    #         print('    Average time per function value, seconds: {}'.
    #                 format(close_timings[0]/close_timings[2]))
    #         print('Total time, seconds: {}'.format(totaltime))
    #         print('Memory:')
    #         print('    Basises, MB: {}'.format(mem[0]))
    #         print('    Transfer matrices, MB: {}'.format(mem[1]))
    #         print('    Far-field interactions, MB: {}'.format(mem[2]))
    #         print('    Near-field interactions, MB: {}'.format(mem[3]))
    #         print('Total memory, MB: {}'.format(mem[4]))
    #         np.set_printoptions(**old_printoptions)
    # if verbose and mpi_comm is None:
    #     print('Far-field interactions(MCBH method):')
    #     print('    Function calls: {}'.format(far_timings[1]))
    #     print('    Function values computed: {}'.format(far_timings[2]))
    #     print('    Function time, seconds: {:.2f}'.format(far_timings[0]))
    #     if far_timings[2] > 0:
    #         print('    Average time per function value, seconds: {:.2e}'.
    #                 format(far_timings[0]/far_timings[2]))
    #     print('    Maxvol time, seconds: {}'.format(far_timings[3]))
    #     print('Near-field interactions:')
    #     print('    Function calls: {}'.format(close_timings[1]))
    #     print('    Function values computed: {}'.format(close_timings[2]))
    #     print('    Function time, seconds: {:.2f}'.format(close_timings[0]))
    #     if close_timings[2] > 0:
    #         print('    Average time per function value, seconds: {:.2e}'.
    #                 format(close_timings[0]/close_timings[2]))
    #     print('Total time, seconds: {:.2f}'.format(totaltime))
    #     print('Memory:')
    #     print('    Basises, MB: {:.2f}'.format(
    #         ans.nbytes(0, 0, 0, 1)/1024/1024))
    #     print('    Transfer matrices, MB: {:.2f}'.format(
    #         ans.nbytes(1, 0, 0, 0)/1024/1024))
    #     print('    Far-field interactions, MB: {:.2f}'.format(
    #         ans.nbytes(0, 1, 0, 0)/1024/1024))
    #     print('    Near-field interactions, MB: {:.2f}'.format(
    #         ans.nbytes(0, 0, 1, 0)/1024/1024))
    #     print('Total memory, MB: {:.2f}'.format(ans.nbytes()/1024/1024))
    # return ans

def _upward_random_node(ind, tree, basis, count):
    """
    Randomly chooses basis for given node hierarchically.
    """
    if tree.child[ind]:
        tmp_index = [basis[j] for j in tree.child[ind]]
        tmp_index = np.concatenate(tmp_index)
    else:
        tmp_index = tree.index[ind]
    return np.random.choice(tmp_index, count)

def _upward_random_sequential(problem, count):
    """
    Sequential random initialization of hierarchical basises.
    """
    tree = problem.tree
    row_notransition = problem.notransition
    tree = problem.tree
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    row_basis = [None for i in range(row_size)]
    if problem.symmetric:
        col_basis = row_basis
    else:
        col_basis = [None for i in range(col_size)]
    for i in range(level_count-1, -1, -1):
        job = [j for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        for ind in job:
            row_basis[ind] = _upward_random_node(ind, tree, row_basis,
                    count)
        if not problem.symmetric:
            job = [j for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            for ind in job:
                col_basis[ind] = _upward_random_node(ind, tree, col_basis,
                        count)
    return row_basis, col_basis

def _upward_random_mpi_static(problem, count, mpi_comm):
    """
    MPI static random initialization of hierarchical basises.
    """
    tree = problem.tree
    row_notransition = problem.notransition
    tree = problem.tree
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
    row_basis = [None for i in range(row_size)]
    if problem.symmetric:
        col_basis = row_basis
    else:
        col_basis = [None for i in range(col_size)]
    for i in range(level_count-1, -1, -1):
        job = [j for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        job_size = len(job)
        my_start = job_size*mpi_rank//mpi_size
        my_end = job_size*(mpi_rank+1)//mpi_size
        result = map(lambda x: _upward_random_node(x, tree, row_basis,\
                count), job[my_start:my_end])
        basis_update = []
        for (ind, res) in zip(job[my_start:my_end], result):
            basis_update.append((ind, res))
        result = mpi_comm.allgather(basis_update)
        total = []
        for subresult in result:
            total.extend(subresult)
        for ind, basis in total:
            row_basis[ind] = basis
        if not problem.symmetric:
            job = [j for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            job_size = len(job)
            my_start = job_size*mpi_rank//mpi_size
            my_end = job_size*(mpi_rank+1)//mpi_size
            result = map(lambda x: _upward_random_node(x, tree, col_basis,\
                    count), job[my_start:my_end])
            basis_update = []
            for (ind, res) in zip(job[my_start:my_end], result):
                basis_update.append((ind, res))
            result = mpi_comm.allgather(basis_update)
            total = []
            for subresult in result:
                total.extend(subresult)
            for ind, basis in total:
                col_basis[ind] = basis
    return row_basis, col_basis

def _upward_random(problem, count, mpi_comm):
    """
    Random initialization of hierarchical basises by `count` elements.
    """
    if mpi_comm is None:
        return _upward_random_sequential(problem, count)
    else:
        return _upward_random_mpi_static(problem, count, mpi_comm)

def _get_basis(tree, basis, prebasis, clusters):
    """
    Unites given prebasis set and basis sets of specified clusters.

    Parameters
    ----------
    tree
        Cluster tree.
    basis
        List-like container of basis sets
    prebasis
        Basis set of cluster parent. Since upward pass computes basis
        sets from bottom to top, this is given from previous iteration
        of MCBH.
    clusters
        List of clusters to unite their basis sets.

    Notes
    -----
    Basis set of a cluster, if not yet computed, is based on basis sets
    of its children. If there are no children, then basis set is equal
    to all cluster elements (particles, discrete elements of surface or
    volume).

    Returns
    -------
    numpy.ndarray
        Concatenation of prebasis set and basis sets of all specified
        clusters.
    """
    if prebasis is None:
        result = []
    else:
        result = [prebasis]
    for i in clusters:
        if basis[i] is None:
            if tree.child[i]:
                for j in tree.child[i]:
                    if basis[j] is None:
                        raise ValueError("Child basises must be defined\
                                before using them. Perhaps, wrong workflow\
                                of program.")
                    result.append(basis[j])
            else:
                result.append(tree.index[i])
        else:
            result.append(basis[i])
    if result:
        return np.concatenate(result)
    else:
        return np.zeros(0, dtype=np.uint64)

def _node_buildmatrix(ind, RC, tree0, far0, tree1, basis0, basis1, prebasis, func, timings):
    """
    Returns basis and interaction matrix of node 'ind' of 'tree0'.
    """
    index0 = _get_basis(tree0, basis0, np.zeros(0, dtype=np.uint64), [ind])
    index1 = _get_basis(tree1, basis1, prebasis, far0[ind])
    time0 = time()
    if RC == 'row':
        matrix = func(index0, index1)
    else:
        matrix = func(index1, index0).T
    timings[0] += time()-time0
    timings[1] += 1
    timings[2] += matrix.size
    return index0, index1, matrix

def _upward_node(j, problem, row_basis, col_basis, row_prebasis, col_prebasis,
        func, timings, tau, alpha, tol):
    """
    Computes transfer matrix and basis for a given node.
    """
    ind = j[1]
    rt = problem.tree
    ct = problem.tree
    rf = problem.far
    cf = problem.far
    rb = row_basis
    cb = col_basis
    rpb = row_prebasis
    cpb = col_prebasis
    if j[0] == 'row':
        basis, tmp, matrix = _node_buildmatrix(ind, j[0], rt, rf, ct, rb, cb,
                rpb[rt.parent[ind]], func, timings)
    else:
        basis, tmp, matrix = _node_buildmatrix(ind, j[0], ct, cf, rt, cb, rb,
                cpb[ct.parent[ind]], func, timings)
    if matrix.size:
        time0 = time()
        matrix = matrix.reshape(matrix.shape[0], -1)
        pivots, result_transfer = maxvol_svd(matrix, tau, alpha, tol, job='R')
        timings[3] += time()-time0
        result_basis = basis[pivots]
    else:
        result_basis = np.zeros(0, dtype=np.uint64)
        result_transfer = np.zeros((basis.size, 0), dtype=matrix.dtype)
    return result_basis, result_transfer

def _upward_sequential(problem, tau, alpha, row_prebasis, col_prebasis,
        timings):
    """
    Sequential upward pass of MCBH method.
    """
    tol = 1.05
    dtype = problem.dtype
    func = problem.func
    tree = problem.tree
    row_far = problem.far
    row_notransition = problem.notransition
    tree = problem.tree
    col_far = problem.far
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    row_basis = [None for i in range(row_size)]
    row_basis_size = np.zeros(row_size, np.uint64)
    row_transfer = [None for i in range(row_size)]
    if problem.symmetric:
        col_basis = row_basis
        col_basis_size = row_basis_size
        col_transfer = row_transfer
    else:
        col_basis = [None for i in range(col_size)]
        col_basis_size = np.zeros(col_size, np.uint64)
        col_transfer = [None for i in range(col_size)]
    for i in range(level_count-1, -1, -1):
        job = [('row', j) for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        for x in job:
            res = _upward_node(x, problem, row_basis, col_basis, row_prebasis,
                    col_prebasis, func, timings, tau, alpha, tol)
            ind = x[1]
            row_basis[ind] = res[0]
            row_basis_size[ind] = res[0].shape[0]
            row_transfer[ind] = res[1]
        if not problem.symmetric:
            job = [('col', j) for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            for x in job:
                res = _upward_node(x, problem, row_basis, col_basis,
                        row_prebasis, col_prebasis, func, timings, tau, alpha,
                        tol)
                ind = x[1]
                col_basis[ind] = res[0]
                col_basis_size[ind] = res[0].shape[0]
                col_transfer[ind] = res[1]
    return row_basis_size, row_basis, row_transfer, col_basis_size, col_basis,\
            col_transfer

def _upward_mpi_static(problem, tau, alpha, row_prebasis, col_prebasis,
        timings, mpi_comm):
    """
    MPI parallel upward pass of MCBH method.
    """
    tol = 1.05
    dtype = problem.dtype
    func = problem.func
    tree = problem.tree
    row_far = problem.far
    row_notransition = problem.notransition
    tree = problem.tree
    col_far = problem.far
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
    row_basis = [None for i in range(row_size)]
    row_basis_size = np.zeros(row_size, np.uint64)
    row_transfer = {}
    if problem.symmetric:
        col_basis = row_basis
        col_basis_size = row_basis_size
        col_transfer = row_transfer
    else:
        col_basis = [None for i in range(col_size)]
        col_basis_size = np.zeros(col_size, np.uint64)
        col_transfer = {}
    for i in range(level_count-1, -1, -1):
        job = [('row', j) for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        job_size = len(job)
        my_start = job_size*mpi_rank//mpi_size
        my_end = job_size*(mpi_rank+1)//mpi_size
        result = map(lambda x: _upward_node(x, problem, row_basis, col_basis,
            row_prebasis, col_prebasis, func, timings, tau, alpha, tol),
            job[my_start:my_end])
        basis_update = []
        for (j, res) in zip(job[my_start:my_end], result):
            ind = j[1]
            row_transfer[ind] = res[1]
            basis_update.append((ind, res[0]))
        result = mpi_comm.allgather(basis_update)
        total = []
        for subresult in result:
            total.extend(subresult)
        for ind, basis in total:
            row_basis[ind] = basis
            row_basis_size[ind] = basis.shape[0]
        if not problem.symmetric:
            job = [('col', j) for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            job_size = len(job)
            my_start = job_size*mpi_rank//mpi_size
            my_end = job_size*(mpi_rank+1)//mpi_size
            result = map(lambda x: _upward_node(x, problem, row_basis,
                col_basis, row_prebasis, col_prebasis, func, timings, tau,
                alpha, tol), job[my_start:my_end])
            basis_update = []
            for (j, res) in zip(job[my_start:my_end], result):
                ind = j[1]
                col_transfer[ind] = res[1]
                basis_update.append((ind, res[0]))
            result = mpi_comm.allgather(basis_update)
            total = []
            for subresult in result:
                total.extend(subresult)
            for ind, basis in total:
                col_basis[ind] = basis
                col_basis_size[ind] = basis.shape[0]
    return row_basis_size, row_basis, row_transfer, col_basis_size, col_basis,\
            col_transfer

def _upward(problem, tau, alpha, row_prebasis, col_prebasis, timings, verbose,
        mpi_comm):
    """
    Wrapper for upward pass of MCBH method.
    """
    if mpi_comm is None:
        return _upward_sequential(problem, tau, alpha, row_prebasis,
                col_prebasis, timings)
    else:
        return _upward_mpi_static(problem, tau, alpha, row_prebasis,
                col_prebasis, timings, mpi_comm)

def _downward_node(j, problem, row_basis, col_basis, row_prebasis,
        col_prebasis, func, timings, tol):
    """
    Computes only basis of predecessor far zone.
    """
    ind = j[1]
    rt = problem.tree
    ct = problem.tree
    rf = problem.far
    cf = problem.far
    rb = row_basis
    cb = col_basis
    rpb = row_prebasis
    cpb = col_prebasis
    if j[0] == 'row':
        tmp, basis, matrix = _node_buildmatrix(ind, j[0], rt, rf, ct, rb, cb,
                rpb[rt.parent[ind]], func, timings)
    else:
        tmp, basis, matrix = _node_buildmatrix(ind, j[0], ct, cf, rt, cb, rb,
                cpb[ct.parent[ind]], func, timings)
    if matrix.size:
        time0 = time()
        matrix = matrix.reshape(-1, matrix.shape[-1])
        pivots = maxvol_qr(matrix.T, tol)[0]
        timings[3] += time()-time0
        result_basis = basis[pivots]
    else:
        result_basis = basis.copy()
    return result_basis

def _downward_sequential(problem, row_basis, col_basis, timings):
    """
    Sequential downward pass of MCBH method.
    """
    tol = 1.05
    dtype = problem.dtype
    func = problem.func
    tree = problem.tree
    row_far = problem.far
    row_notransition = problem.notransition
    tree = problem.tree
    col_far = problem.far
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    row_prebasis = [None for i in range(row_size)]
    if problem.symmetric:
        col_prebasis = row_prebasis
    else:
        col_prebasis = [None for i in range(col_size)]
    for i in range(level_count-1, -1, -1):
        job = [('row', j) for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        result = map(lambda x: _downward_node(x, problem, row_basis, col_basis,
            row_prebasis, col_prebasis, func, timings, tol), job)
        for (j, res) in zip(job, result):
            ind = j[1]
            row_prebasis[ind] = res
        if not problem.symmetric:
            job = [('col', j) for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            result = map(lambda x: _downward_node(x, problem, row_basis,
                col_basis, row_prebasis, col_prebasis, func, timings, tol),
                job)
            for (j, res) in zip(job, result):
                ind = j[1]
                col_prebasis[ind] = res
    return row_prebasis, col_prebasis

def _downward_mpi_static(problem, row_basis, col_basis, timings, mpi_comm):
    """
    MPI parallel downward pass of MCBH method.
    """
    tol = 1.05
    dtype = problem.dtype
    func = problem.func
    tree = problem.tree
    row_far = problem.far
    row_notransition = problem.notransition
    tree = problem.tree
    col_far = problem.far
    col_notransition = problem.notransition
    row_size = tree.level[-1]
    col_size = tree.level[-1]
    level_count = len(tree.level)-2
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
    row_prebasis = [None for i in range(row_size)]
    if problem.symmetric:
        col_prebasis = row_prebasis
    else:
        col_prebasis = [None for i in range(col_size)]
    for i in range(level_count-1, -1, -1):
        job = [('row', j) for j in
                range(tree.level[i], tree.level[i+1])
                if not row_notransition[j]
                ]
        job_size = len(job)
        my_start = job_size*mpi_rank//mpi_size
        my_end = job_size*(mpi_rank+1)//mpi_size
        result = map(lambda x: _downward_node(x, problem, row_basis, col_basis,
            row_prebasis, col_prebasis, func, timings, tol),
            job[my_start:my_end])
        basis_update = []
        for (j, res) in zip(job[my_start:my_end], result):
            ind = j[1]
            row_prebasis[ind] = res
            basis_update.append((ind, res))
        result = mpi_comm.allgather(basis_update)
        total = []
        for subresult in result:
            total.extend(subresult)
        for ind, basis in total:
            row_prebasis[ind] = basis
        if not problem.symmetric:
            job = [('col', j) for j in
                    range(tree.level[i], tree.level[i+1])
                    if not col_notransition[j]
                    ]
            job_size = len(job)
            my_start = job_size*mpi_rank//mpi_size
            my_end = job_size*(mpi_rank+1)//mpi_size
            result = map(lambda x: _downward_node(x, problem, row_basis,
                col_basis, row_prebasis, col_prebasis, func, timings,
                tol), job[my_start:my_end])
            basis_update = []
            for (j, res) in zip(job[my_start:my_end], result):
                ind = j[1]
                col_prebasis[ind] = res
                basis_update.append((ind, res))
            result = mpi_comm.allgather(basis_update)
            total = []
            for subresult in result:
                total.extend(subresult)
            for ind, basis in total:
                col_prebasis[ind] = basis
    return row_prebasis, col_prebasis

def _downward(problem, row_prebasis, col_prebasis, timings, verbose, mpi_comm):
    """
    Wrapper for downward pass of MCBH method.
    """
    if mpi_comm is None:
        return _downward_sequential(problem, row_prebasis, col_prebasis,
                timings)
    else:
        return _downward_mpi_static(problem, row_prebasis, col_prebasis,
                timings, mpi_comm)
