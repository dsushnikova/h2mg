"""
Class H2matrix represents H2-matrices in special low-parametric format.
"""

from __future__ import print_function, absolute_import, division

from time import time
import numpy as np
from .maxvol import maxvol_svd, maxvol_qr
from .problem import Problem
import copy
try:
    from pypropack import svdp
    from mpi4py import MPI
except:
    pass
import scipy.sparse.linalg as la
from sys import getsizeof
from .mpi_misc import sync_mpi_data

__all__ = ['H2matrix']

class H2matrix(object):
    def __init__(self, problem, row_basis_size, col_basis_size, row_transfer, col_transfer, row_interaction, col_interaction, row_close, col_close, row_basis=None, col_basis=None, mpi_comm=None):
        if not isinstance(problem, Problem):
            raise TypeError('problem parameter must be instance of'
                    ' h2tools.Problem')
        if not (((row_basis is None) and (col_basis is None)) or
                (isinstance(row_basis, list) and isinstance(col_basis, list))):
            raise TypeError('row_basis and col_basis parameters should be'
                    ' both None or both lists of corresponding basises.')
        if row_interaction is None and col_interaction is not None:
            try:
                size = problem.row_tree.level[-1]
                row_interaction = [[] for i in range(size)]
                for i in range(size):
                    for j in problem.row_far[i]:
                        k = problem.col_far[j].index(i)
                        row_interaction[i].append(col_interaction[j][k].T)
            except:
                raise ValueError('col_interaction must be None or a list of'
                        ' lists of far-field interaction matrices of'
                        ' appropriate size.')
        if col_interaction is None and row_interaction is not None:
            try:
                size = problem.col_tree.level[-1]
                col_interaction = [[] for i in range(size)]
                for i in range(size):
                    for j in problem.col_far[i]:
                        k = problem.row_far[j].index(i)
                        col_interaction[i].append(row_interaction[j][k].T)
            except:
                raise ValueError('row_interaction must be None or a list of'
                        ' lists of far-field interaction matrices of'
                        ' appropriate size.')
        if row_close is None and col_close is not None:
            try:
                size = problem.row_tree.level[-1]
                row_close = [[] for i in range(size)]
                for i in range(size):
                    for j in problem.row_close[i]:
                        k = problem.col_close[j].index(i)
                        row_close[i].append(col_close[j][k].T)
            except:
                raise ValueError('col_close must be None or a list of lists'
                        ' of near-field interaction matrices of appropriate'
                        ' size.')
        if col_close is None and row_close is not None:
            try:
                size = problem.col_tree.level[-1]
                col_close = [[] for i in range(size)]
                for i in range(size):
                    for j in problem.col_close[i]:
                        k = problem.row_close[j].index(i)
                        col_close[i].append(row_close[j][k].T)
            except:
                raise ValueError('row_close must be None or a list of lists'
                        ' of near-field interaction matrices of appropriate'
                        ' size.')
        if row_basis is None:
            self.h2_type = 'h2'
        else:
            self.h2_type = 'mcbh'
        if self.h2_type == 'mcbh' and ((row_close is None) is
                (row_interaction is not None)):
            print('Since only one of far-field and close-field interactions'
                    ' were assigned as None, assuming "low" memory allocation'
                    ' type (both set as None).')
            row_interaction = None
            col_interaction = None
            row_close = None
            col_close = None
        if row_close is None:
            self.mem_type = 'low'
        else:
            self.mem_type = 'full'
        self.problem = problem
        self.shape = problem.shape
        self.element_shape = problem.func_shape
        self.dtype = problem.dtype
        self.row_basis_size = row_basis_size
        self.col_basis_size = col_basis_size
        self.row_basis = row_basis
        self.col_basis = col_basis
        self.row_transfer = row_transfer
        self.col_transfer = col_transfer
        self.row_interaction = row_interaction
        self.col_interaction = col_interaction
        self.row_close = row_close
        self.col_close = col_close
        self.mpi_comm = mpi_comm

    def func(self, row, col):
        """
        Returns submatrix on intersection of given rows and columns.

        Parameters
        ----------
        row : numpy.ndarray(ndim=1, dtype=numpy.uint64)
            rows, where to compute submatrix
        col : numpy.ndarray(ndim=1, dtype=numpy.uint64)
            columns, where to compute submatrix

        Returns
        -------
        numpy.ndarray
            Submatrix on intersection of given rows and columns.
        """
        return self.problem.func(row, col)

    def _matrix_sequential(self, timings):
        """
        Sequential computation of far-field interaction matrices.
        """
        func = self.problem.func
        row = self.problem.row_tree
        row_far = self.problem.row_far
        col = self.problem.col_tree
        col_far = self.problem.col_far
        row_basis = self.row_basis
        col_basis = self.col_basis
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_interaction = [[None for j in row_far[i]] for i in range(row_size)]
        symmetric = self.problem.symmetric
        for i in range(row_size):
            for j in range(len(row_far[i])):
                if row_interaction[i][j] is None:
                    k = row_far[i][j]
                    time0 = time()
                    tmp_matrix = func(row_basis[i], col_basis[k])
                    if timings is not None:
                        timings[0] += time()-time0
                        timings[1] += 1
                        timings[2] += tmp_matrix.size
                    row_interaction[i][j] = tmp_matrix
                    if symmetric:
                        l = row_far[k].index(i)
                        row_interaction[k][l] = tmp_matrix.T
        if symmetric:
            col_interaction = row_interaction
        else:
            col_interaction = [[None for j in col_far[i]] for i in
                    range(col_size)]
            for i in range(col_size):
                for j in range(len(col_far[i])):
                    jj = col_far[i][j]
                    k = row_far[jj].index(i)
                    if row_interaction[jj][k] is not None:
                        col_interaction[i][j] = row_interaction[jj][k].T
        self.row_interaction = row_interaction
        self.col_interaction = col_interaction

    def _matrix_mpi_static(self, timings):
        """
        MPI parallel computations of far-field interaction matrices.
        """
        mpi_comm = self.mpi_comm
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        func = self.problem.func
        row = self.problem.row_tree
        row_far = self.problem.row_far
        col = self.problem.col_tree
        col_far = self.problem.col_far
        row_basis = self.row_basis
        col_basis = self.col_basis
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_interaction = {}
        symmetric = self.problem.symmetric
        if not symmetric:
            col_interaction = {}
            for i in range(mpi_rank, row_size, mpi_size):
                if row_far[i]:
                    row_interaction[i] = {}
                for j in row_far[i]:
                    time0 = time()
                    tmp_matrix = func(row_basis[i], col_basis[j])
                    if timings is not None:
                        timings[0] += time()-time0
                        timings[1] += 1
                        timings[2] += tmp_matrix.size
                    row_interaction[i][j] = tmp_matrix
                    if j not in col_interaction:
                        col_interaction[j] = {}
                    col_interaction[j][i] = tmp_matrix.T
        else:
            col_interaction = row_interaction
            for i in range(mpi_rank, row_size, mpi_size):
                for j in row_far[i]:
                    if j >= i:
                        time0 = time()
                        tmp_matrix = func(row_basis[i], col_basis[j])
                        if timings is not None:
                            timings[0] += time()-time0
                            timings[1] += 1
                            timings[2] += tmp_matrix.size
                        if i not in row_interaction:
                            row_interaction[i] = {}
                        row_interaction[i][j] = tmp_matrix
                        if j not in row_interaction:
                            row_interaction[j] = {}
                        row_interaction[j][i] = tmp_matrix.T
        self.row_interaction = row_interaction
        self.col_interaction = col_interaction

    def _matrix(self, timings=None):
        """
        Wrapper for computations of far-field interaction matrices.
        """
        if self.mpi_comm is None:
            self._matrix_sequential(timings)
        else:
            self._matrix_mpi_static(timings)

    def _close_matrix_sequential(self, timings):
        """
        Sequential computations of near-field interaction matrices.
        """
        func = self.problem.func
        row = self.problem.row_tree
        row_close = self.problem.row_close
        col = self.problem.col_tree
        col_close = self.problem.col_close
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_close_interaction = [[None for j in row_close[i]]
                for i in range(row_size)]
        symmetric = self.problem.symmetric
        for i in range(row_size):
            for j in range(len(row_close[i])):
                if row_close_interaction[i][j] is None:
                    k = row_close[i][j]
                    time0 = time()
                    tmp_matrix = func(row.index[i], col.index[k])
                    if timings is not None:
                        timings[0] += time()-time0
                        timings[1] += 1
                        timings[2] += tmp_matrix.size
                    row_close_interaction[i][j] = tmp_matrix
                    if symmetric and k != i:
                        l = row_close[k].index(i)
                        row_close_interaction[k][l] = tmp_matrix.T
        if symmetric:
            col_close_interaction = row_close_interaction
        else:
            col_close_interaction = [[None for j in col_close[i]]
                    for i in range(col_size)]
            for i in range(col_size):
                for j in range(len(col_close[i])):
                    jj = col_close[i][j]
                    k = row_close[jj].index(i)
                    if row_close_interaction[jj][k] is not None:
                        col_close_interaction[i][j] =\
                                row_close_interaction[jj][k].T
        self.row_close = row_close_interaction
        self.col_close = col_close_interaction

    def _close_matrix_mpi_static(self, timings):
        """
        MPI parallel computations of near-field interaction matrices.
        """
        mpi_comm = self.mpi_comm
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        func = self.problem.func
        row = self.problem.row_tree
        row_close = self.problem.row_close
        col = self.problem.col_tree
        col_close = self.problem.col_close
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_close_interaction = {}
        symmetric = self.problem.symmetric
        if not symmetric:
            col_close_interaction = {}
            for i in range(mpi_rank, row_size, mpi_size):
                for j in row_close[i]:
                    time0 = time()
                    tmp_matrix = func(row.index[i], col.index[j])
                    if timings is not None:
                        timings[0] += time()-time0
                        timings[1] += 1
                        timings[2] += tmp_matrix.size
                    if i not in row_close_interaction:
                        row_close_interaction[i] = {}
                    row_close_interaction[i][j] = tmp_matrix
                    if j not in col_close_interaction:
                        col_close_interaction[j] = {}
                    col_close_interaction[j][i] = tmp_matrix.T
        else:
            col_close_interaction = row_close_interaction
            for i in range(mpi_rank, row_size, mpi_size):
                for j in row_close[i]:
                    if j >= i:
                        time0 = time()
                        tmp_matrix = func(row.index[i], col.index[j])
                        if timings is not None:
                            timings[0] += time()-time0
                            timings[1] += 1
                            timings[2] += tmp_matrix.size
                        if i not in row_close_interaction:
                            row_close_interaction[i] = {}
                        row_close_interaction[i][j] = tmp_matrix
                    if j > i:
                        if j not in row_close_interaction:
                            row_close_interaction[j] = {}
                        row_close_interaction[j][i] = tmp_matrix.T
        self.row_close = row_close_interaction
        self.col_close = col_close_interaction

    def _close_matrix(self, timings=None):
        """
        Wrapper for computations of near-field interaction matrices.
        """
        if self.mpi_comm is None:
            self._close_matrix_sequential(timings)
        else:
            self._close_matrix_mpi_static(timings)

    def _dot_up(self, tree, transfer, x, mpi_comm):
        """
        Computes "charges" of basis elements.

        Does it in bottom-to-top manner.
        """
        size = tree.level[-1]
        node_weight = [None for i in range(size)]
        if mpi_comm is None:
            self._dot_up_sequential(tree, transfer, x, node_weight)
        else:
            self._dot_up_mpi_static(tree, transfer, x, node_weight, mpi_comm)
        return node_weight

    def _dot_up_sequential(self, tree, transfer, x, node_weight):
        """
        Sequential `_dot_up`.
        """
        level_count = len(tree.level)-2
        for i in range(level_count-1, -1, -1):
            level = range(tree.level[i], tree.level[i+1])
            for j in level:
                if transfer[j] is None or transfer[j].shape[1] == 0:
                    continue
                if not tree.child[j]:
                    tmp = x[tree.index[j]]
                else:
                    tmp = []
                    for k in tree.child[j]:
                        if node_weight[k] is not None:
                            tmp.append(node_weight[k])
                    tmp = np.concatenate(tmp, axis=0)
                node_weight[j] = np.tensordot(transfer[j].T, tmp, 1)

    def _dot_up_mpi_static(self, tree, transfer, x, node_weight, mpi_comm):
        """
        MPI parallel `_dot_up`.
        """
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        level_count = len(tree.level)-2
        my_index = np.array(transfer.keys())
        my_index.sort()
        level_division = my_index.searchsorted(tree.level)
        for i in range(level_count-1, -1, -1):
            level = my_index[level_division[i]:level_division[i+1]]
            result = []
            for j in level:
                if transfer[j] is None or transfer[j].shape[1] == 0:
                    continue
                if not tree.child[j]:
                    tmp = x[tree.index[j]]
                else:
                    tmp = []
                    for k in tree.child[j]:
                        if node_weight[k] is not None:
                            tmp.append(node_weight[k])
                    tmp = np.concatenate(tmp, axis=0)
                tmp = np.tensordot(transfer[j].T, tmp, 1)
                result.append((j, tmp))
            result = mpi_comm.allgather(result)
            total = []
            for subresult in result:
                total.extend(subresult)
            for j in total:
                node_weight[j[0]] = j[1]

    def _dot_down(self, tree, transfer, basis, node_answer, answer, mpi_comm):
        """
        Computes "potentials" of all elements.

        Uses precomputed "potentials" of cluster tree.
        """
        if mpi_comm is None:
            self._dot_down_sequential(tree, transfer, basis, node_answer,
                    answer)
        else:
            self._dot_down_mpi_static(tree, transfer, basis, node_answer,
                    answer, mpi_comm)
        return

    def _dot_down_sequential(self, tree, transfer, basis_size, node_answer, answer):
        """
        Sequential `_dot_down`.
        """
        size = tree.level[-1]
        level_count = len(tree.level)-2
        for i in range(level_count):
            level = range(tree.level[i], tree.level[i+1])
            for j in level:
                if transfer[j] is None or transfer[j].size == 0:
                    continue
                tmp = np.tensordot(transfer[j], node_answer[j], 1)
                if tree.child[j]:
                    i1 = 0
                    for k in tree.child[j]:
                        i2 = i1+basis_size[k]
                        if node_answer[k] is None:
                            node_answer[k] = tmp[i1:i2]
                        else:
                            node_answer[k] += tmp[i1:i2]
                        i1 = i2
                else:
                    answer[tree.index[j]] = tmp

    def _dot_down_mpi_static(self, tree, transfer, basis_size, node_answer, answer, mpi_comm):
        """
        MPI parallel `_dot_down`.
        """
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        level_count = len(tree.level)-2
        my_index = np.array(transfer.keys())
        my_index.sort()
        level_division = my_index.searchsorted(tree.level)
        lanswer = np.zeros(answer.shape, dtype=answer.dtype)
        for i in range(level_count):
            result = []
            level = my_index[level_division[i]:level_division[i+1]]
            for j in level:
                if transfer[j] is None or transfer[j].size == 0:
                    continue
                tmp = np.tensordot(transfer[j], node_answer[j], 1)
                if tree.child[j]:
                    i1 = 0
                    for k in tree.child[j]:
                        i2 = i1+basis_size[k]
                        if node_answer[k] is None:
                            result.append((k, tmp[i1:i2]))
                        else:
                            result.append((k, node_answer[k]+tmp[i1:i2]))
                        i1 = i2
                else:
                    lanswer[tree.index[j]] = tmp
            result = mpi_comm.allgather(result)
            total = []
            for subresult in result:
                total.extend(subresult)
            for j in total:
                node_answer[j[0]] = j[1]
        mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)

    def _dot_interact(self, tree, far, interaction, node_weight, mpi_comm):
        """
        Computes "potentials" of basis elements.

        Uses precomputed interaction matrices and "charges" of adjunct
        cluster tree.
        """
        if mpi_comm is None:
            return self._dot_interact_sequential(tree, far, interaction,
                    node_weight)
        else:
            return self._dot_interact_mpi_static(tree, far, interaction,
                    node_weight, mpi_comm)

    def _dot_interact_sequential(self, tree, far, interaction, node_weight):
        """
        Sequential `_dot_interact`.
        """
        size = tree.level[-1]
        node_answer = [None for i in range(size)]
        for i in range(size):
            if far[i]:
                tmp = None
                for j in range(len(far[i])):
                    tmp2 = np.tensordot(interaction[i][j],
                            node_weight[far[i][j]], 1)
                    if tmp is None:
                        tmp = tmp2
                    else:
                        tmp += tmp2
                node_answer[i] = tmp
        return node_answer

    def _dot_interact_mpi_static(self, tree, far, interaction, node_weight, mpi_comm):
        """
        MPI parallel `_dot_interact`.
        """
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        size = tree.level[-1]
        node_answer = [None for i in range(size)]
        local_data = {}
        for i, interaction2 in interaction.iteritems():
            tmp = None
            for j, matrix in interaction2.iteritems():
                if tmp is None:
                    tmp = np.tensordot(matrix, node_weight[j], 1)
                else:
                    tmp += np.tensordot(matrix, node_weight[j], 1)
            if tmp is not None:
                local_data[i] = tmp
        sync_mpi_data(local_data, node_answer, mpi_comm)
        return node_answer

    def _dot_interact_onfly(self, tree, far, func, basis0, basis1, node_weight, T=False, mpi_comm=None):
        """
        Computes "potentials" of basis elements.

        Computes interaction matrices on fly and uses precomputed
        "charges" of adjunct cluster tree.
        """
        if mpi_comm is None:
            return self._dot_interact_onfly_sequential(tree, far, func,
                    basis0, basis1, node_weight, T)
        else:
            return self._dot_interact_onfly_mpi_static(tree, far, func,
                    basis0, basis1, node_weight, T, mpi_comm)

    def _dot_interact_onfly_sequential(self, tree, far, func, basis0, basis1, node_weight, T=False):
        """
        Sequential `_dot_interact_onfly`.
        """
        size = tree.level[-1]
        node_answer = [None for i in range(size)]
        if self.problem.symmetric:
            for i in range(size):
                for j in far[i]:
                    if i <= j:
                        if T:
                            tmp_matrix = func(basis1[j], basis0[i]).T
                        else:
                            tmp_matrix = func(basis0[i], basis1[j])
                        tmp = np.tensordot(tmp_matrix, node_weight[j], 1)
                        if node_answer[i] is None:
                            node_answer[i] = tmp
                        else:
                            node_answer[i] += tmp
                        tmp = np.tensordot(tmp_matrix.T, node_weight[i], 1)
                        if node_answer[j] is None:
                            node_answer[j] = tmp
                        else:
                            node_answer[j] += tmp
        else:
            for i in range(size):
                for j in far[i]:
                    if T:
                        tmp_matrix = func(basis1[j], basis0[i]).T
                    else:
                        tmp_matrix = func(basis0[i], basis1[j])
                    tmp = np.tensordot(tmp_matrix, node_weight[j], 1)
                    if node_answer[i] is None:
                        node_answer[i] = tmp
                    else:
                        node_answer[i] += tmp
        return node_answer

    def _dot_interact_onfly_mpi_static(self, tree, far, func, basis0, basis1, node_weight, T=False, mpi_comm=None):
        """
        MPI parallel `_dot_interact_onfly`.
        """
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        size = tree.level[-1]
        node_answer = [None for i in range(size)]
        t0 = time()
        if self.problem.symmetric:
            local_data = {}
            for i in range(mpi_rank, size, mpi_size):
                for j in far[i]:
                    if i <= j:
                        if T:
                            tmp_matrix = func(basis1[j], basis0[i]).T
                        else:
                            tmp_matrix = func(basis0[i], basis1[j])
                        tmp = np.tensordot(tmp_matrix, node_weight[j], 1)
                        if i not in local_data:
                            local_data[i] = tmp
                        else:
                            local_data[i] += tmp
                        tmp = np.tensordot(tmp_matrix.T, node_weight[i], 1)
                        if j not in local_data:
                            local_data[j] = tmp
                        else:
                            local_data[j] += tmp
            sync_mpi_data(local_data, node_answer, mpi_comm)
        else:
            result = []
            for i in range(mpi_rank, size, mpi_size):
                for j in far[i]:
                    if T:
                        tmp_matrix = func(basis1[j], basis0[i]).T
                    else:
                        tmp_matrix = func(basis0[i], basis1[j])
                    tmp = np.tensordot(tmp_matrix, node_weight[j], 1)
                    if node_answer[i] is None:
                        node_answer[i] = tmp
                    else:
                        node_answer[i] += tmp
                if node_answer[i] is not None:
                    result.append((i, node_answer[i]))
            result = mpi_comm.allgather(result)
            for i in result:
                for j in i:
                    node_answer[j[0]] = j[1]
        return node_answer

    def far_dot(self, x):
        """
        Applies far-field part of H2matrix to vector from left side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of farfield `A * x`.
        """
        if x.shape[0] != self.shape[1]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[1], x.shape[0])
        node_weight = self._dot_up(self.problem.col_tree, self.col_transfer, x,
                self.mpi_comm)
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            node_answer = self._dot_interact_onfly(self.problem.row_tree,
                    self.problem.row_far, self.problem.func, self.row_basis,
                    self.col_basis, node_weight, False, self.mpi_comm)
        else:
            node_answer = self._dot_interact(self.problem.row_tree,
                    self.problem.row_far, self.row_interaction, node_weight,
                    self.mpi_comm)
        answer_shape = [self.shape[0]]
        answer_shape.extend(self.element_shape)
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        self._dot_down(self.problem.row_tree, self.row_transfer,
                self.row_basis_size, node_answer, answer, self.mpi_comm)
        return answer

    def far_rdot(self, x):
        """
        Applies far-field part of H2matrix to vector from right side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of farfield `A.T * x`.
        """
        if x.shape[0] != self.shape[0]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[0], x.shape[0])
        node_weight = self._dot_up(self.problem.row_tree, self.row_transfer, x,
                self.mpi_comm)
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            node_answer = self._dot_interact_onfly(self.problem.col_tree,
                    self.problem.col_far, self.problem.func, self.col_basis,
                    self.row_basis, node_weight, True, self.mpi_comm)
        else:
            node_answer = self._dot_interact(self.problem.col_tree,
                    self.problem.col_far, self.col_interaction, node_weight,
                    self.mpi_comm)
        answer_shape = [self.shape[1]]
        answer_shape.extend(reversed(self.element_shape))
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        self._dot_down(self.problem.col_tree, self.col_transfer,
                self.col_basis_size, node_answer, answer, self.mpi_comm)
        return answer

    def _close_dot(self, tree0, tree1, close, interaction, x, answer, mpi_comm):
        """
        Wrapper for near-field interactions.
        """
        if mpi_comm is None:
            self._close_dot_sequential(tree0, tree1, close, interaction, x,
                    answer)
        else:
            self._close_dot_mpi_static(tree0, tree1, close, interaction, x,
                    answer, mpi_comm)

    def _close_dot_sequential(self, tree0, tree1, close, interaction, x, answer):
        """
        Sequential near-field interactions.
        """
        size = tree0.level[-1]
        for i in range(size):
            for j in range(len(close[i])):
                answer[tree0.index[i]] += interaction[i][j].dot(x[tree1.index[
                    close[i][j]]])

    def _close_dot_mpi_static(self, tree0, tree1, close, interaction, x, answer, mpi_comm):
        """
        MPI parallel near-field interactions.
        """
        lanswer = np.zeros(answer.shape, dtype=answer.dtype)
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        size = tree0.level[-1]
        for i, interaction2 in interaction.iteritems():
            tmp = lanswer[tree0.index[i]]
            for j, matrix in interaction2.iteritems():
                tmp += np.tensordot(matrix, x[tree1.index[j]], 1)
            lanswer[tree0.index[i]] = tmp
        mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)

    def _close_dot_onfly(self, tree0, tree1, close, func, x, answer, T, mpi_comm):
        """
        Wrapper for onfly near-field interactions.
        """
        if mpi_comm is None:
            self._close_dot_onfly_sequential(tree0, tree1, close, func, x,
                    answer, T)
        else:
            self._close_dot_onfly_mpi_static(tree0, tree1, close, func, x,
                    answer, T, mpi_comm)

    def _close_dot_onfly_sequential(self, tree0, tree1, close, func, x, answer, T):
        """
        Sequential onfly near-field interactions.
        """
        size = tree0.level[-1]
        if self.problem.symmetric:
            for i in range(size):
                ind0 = tree0.index[i]
                for j in close[i]:
                    if i <= j:
                        ind1 = tree1.index[j]
                        if T:
                            tmp_matrix = func(ind1, ind0).T
                        else:
                            tmp_matrix = func(ind0, ind1)
                        answer[ind0] += np.tensordot(tmp_matrix, x[ind1], 1)
                    if i < j:
                        answer[ind1] += np.tensordot(tmp_matrix.T, x[ind0], 1)
        else:
            for i in range(size):
                ind0 = tree0.index[i]
                for j in close[i]:
                    ind1 = tree1.index[j]
                    if T:
                        tmp_matrix = func(ind1, ind0).T
                    else:
                        tmp_matrix = func(ind0, ind1)
                    answer[ind0] += np.tensordot(tmp_matrix, x[ind1], 1)

    def _close_dot_onfly_mpi_static(self, tree0, tree1, close, func, x, answer, T, mpi_comm):
        """
        MPI parallel onfly near-field interactions.
        """
        lanswer = np.zeros(answer.shape, dtype=answer.dtype)
        mpi_rank = mpi_comm.rank
        mpi_size = mpi_comm.size
        size = tree0.level[-1]
        basis0 = tree0.index
        basis1 = tree1.index
        if self.problem.symmetric:
            for i in range(mpi_rank, size, mpi_size):
                ind0 = tree0.index[i]
                for j in close[i]:
                    if i <= j:
                        ind1 = tree1.index[j]
                        if T:
                            tmp_matrix = func(ind1, ind0).T
                        else:
                            tmp_matrix = func(ind0, ind1)
                        lanswer[ind0] += np.tensordot(tmp_matrix, x[ind1], 1)
                    if i < j:
                        lanswer[ind1] += np.tensordot(tmp_matrix.T, x[ind0], 1)
        else:
            for i in range(mpi_rank, size, mpi_size):
                ind0 = tree0.index[i]
                for j in close[i]:
                    ind1 = tree1.index[j]
                    if T:
                        tmp_matrix = func(ind1, ind0).T
                    else:
                        tmp_matrix = func(ind0, ind1)
                    lanswer[ind0] += np.tensordot(tmp_matrix, x[ind1], 1)
        mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)

    def close_dot(self, x):
        """
        Applies near-field part of H2matrix to vector from left side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of near-field `A * x`.
        """
        if x.shape[0] != self.shape[1]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[1], x.shape[0])
        func = self.problem.func
        row = self.problem.row_tree
        col = self.problem.col_tree
        close = self.problem.row_close
        answer_shape = [self.shape[0]]
        answer_shape.extend(self.element_shape)
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            self._close_dot_onfly(row, col, close, func, x, answer, False,
                    self.mpi_comm)
        else:
            self._close_dot(row, col, close, self.row_close, x, answer,
                    self.mpi_comm)
        return answer

    def close_rdot(self, x):
        """
        Applies near-field part of H2matrix to vector from right side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of near-field `A.T * x`.
        """
        if x.shape[0] != self.shape[0]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[0], x.shape[0])
        func = self.problem.func
        row = self.problem.row_tree
        col = self.problem.col_tree
        close = self.problem.col_close
        answer_shape = [self.shape[1]]
        answer_shape.extend(reversed(self.element_shape))
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            self._close_dot_onfly(col, row, close, func, x, answer, True,
                    self.mpi_comm)
        else:
            self._close_dot(col, row, close, self.col_close, x, answer,
                    self.mpi_comm)
        return answer

    def dot(self, x):
        """
        Applies multplication by H2matrix to vector from left side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of `A * x`.
        """
        if x.shape[0] != self.shape[1]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[1], x.shape[0])
        answer = self.far_dot(x)+self.close_dot(x)
        return answer

    def rdot(self, x):
        """
        Applies multplication by H2matrix to vector from right side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of `A.T * x`.
        """
        if x.shape[0] != self.shape[0]:
            raise ValueError('operands could not be broadcast together with'
                    ' shapes ({0:d}) ({1:d})').format(
                            self.shape[0], x.shape[0])
        answer = self.far_rdot(x)+self.close_rdot(x)
        return answer

    def nbytes(self, transfer=True, interaction=True, close=True, basis=True):
        """
        Returns memory, used by requested parts of H2matrix, in bytes.

        Parameters
        ----------
        transfer : boolean
            If true, adds to result size of memory buffers, used by
            transfer matrices.
        interaction : boolean
            If true, adds to result size of memory buffers, used by
            far-field interaction matrices.
        close : boolean
            If true, adds to result size of memory buffers, used by
            near-field interaction matrices.
        basis : boolean
            If true, adds to result size of memory buffers, used by
            storage of indexes of basis rows and columns.

        Returns
        -------
        integer
            Number of bytes, consumed by given parts of H2matrix.
        """
        nbytes = getsizeof(self)
        if transfer:
            nbytes += getsizeof(self.row_transfer)
            if isinstance(self.row_transfer, dict):
                for i in self.row_transfer.itervalues():
                    nbytes += getsizeof(i)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_transfer)
                    for i in self.col_transfer.itervalues():
                        nbytes += getsizeof(i)
            else:
                for i in self.row_transfer:
                    nbytes += getsizeof(i)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_transfer)
                    for i in self.col_transfer:
                        nbytes += getsizeof(i)
        if interaction and not (self.mem_type == 'low' and
                self.h2_type == 'mcbh'):
            nbytes += getsizeof(self.row_interaction)
            if isinstance(self.row_interaction, dict):
                for i in self.row_interaction.itervalues():
                    nbytes += getsizeof(i)
                    for j in i.itervalues():
                        nbytes += getsizeof(j)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_interaction)
                    for i in self.col_interaction.itervalues():
                        nbytes += getsizeof(i)
                        for j in i.itervalues():
                            nbytes += getsizeof(j)
            else:
                for i in self.row_interaction:
                    nbytes += getsizeof(i)
                    for j in i:
                        nbytes += getsizeof(j)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_interaction)
                    for i in self.col_interaction:
                        nbytes += getsizeof(i)
                        for j in i:
                            nbytes += getsizeof(j)
        if close and not self.mem_type == 'low':
            nbytes += getsizeof(self.row_close)
            if isinstance(self.row_close, dict):
                for i in self.row_close.itervalues():
                    nbytes += getsizeof(i)
                    for j in i.itervalues():
                        nbytes += getsizeof(j)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_close)
                    for i in self.col_close.itervalues():
                        nbytes += getsizeof(i)
                        for j in i.itervalues():
                            nbytes += getsizeof(j)
            else:
                for i in self.row_close:
                    nbytes += getsizeof(i)
                    for j in i:
                        nbytes += getsizeof(j)
                if not self.problem.symmetric:
                    nbytes += getsizeof(self.col_close)
                    for i in self.col_close:
                        nbytes += getsizeof(i)
                        for j in i:
                            nbytes += getsizeof(j)
        if basis and self.h2_type == 'mcbh':
            nbytes += getsizeof(self.row_basis)
            for i in self.row_basis:
                nbytes += getsizeof(i)
            if not self.problem.symmetric:
                nbytes += getsizeof(self.col_basis)
                for i in self.col_basis:
                    nbytes += getsizeof(i)
        if basis:
            nbytes += getsizeof(self.row_basis_size)
            if not self.problem.symmetric:
                nbytes += getsizeof(self.col_basis_size)
        return nbytes

    def svdcompress(self, tau, verbose=False):
        """
        Decompression of H2matrix by SVD for block rows and columns.

        Performs SVD decompression of each block row and block column
        inplace. Orthogonolizes transfer matrices of row cluster tree,
        then decompresses block columns of column cluster tree, then
        orthogonolizes transfer matrices of column cluster tree and
        then decompresses block rows of row cluster tree. Finally,
        transfer matrices of row cluster tree are orthogonolized one
        additional time to make all transfer matrices orthogonal.

        Parameters
        ----------
        tau : float
            Spectral error tolerance for SVD decompressions of each block
            row and block column.
        verbose : boolean
            If true, shows memory before and after decompression and
            additional information in some cases.

        Notes
        -----
        Works only in sequential programs
        """
        if self.mpi_comm is not None:
            raise NotImplementedError("MPI parallel compression is not yet"
                    " implemented.")
        time0 = time()
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            if verbose:
                print('Computing far-field interaction matrices, since they'
                        ' are required for svd decompression')
            self._matrix()
        if verbose:
            print('memory BEFORE SVD-compression: {0:.2f}MB'.format(
                self.nbytes()/1024./1024))
        if self.problem.symmetric:
            # if symmetry flag is True, then each item of self.queue
            # contains only 'row' tag
            self._orthogonolize('row')
            self._compress('row', tau)
        else:
            self._orthogonolize('row')
            self._compress('col', tau)
            self._compress('row', tau)
        self._compresstime = time()-time0
        self.row_basis = None
        self.col_basis = None
        self.h2_type = 'h2'
        if verbose:
            print('memory AFTER SVD-compression: {0:.2f}MB'.format(
                self.nbytes()/1024./1024))
            print('recompression time:', self._compresstime)

    def _orthogonolize(self, RC):
        """
        Orthogonolizes transfer matrices for given cluster tree.

        Parameters
        ----------
        RC : string
            Shows what cluster tree's transfer matrices need to be
            orthogonolized. Possible values are "row" and "col".

        Notes
        -----
        Works only in sequential programs.
        """
        if RC == 'row':
            tree = self.problem.row_tree
            far = self.problem.row_far
            far2 = self.problem.col_far
            transfer = self.row_transfer
            interaction = self.row_interaction
            interaction2 = self.col_interaction
            node_count = tree.level[-1]
        else:
            tree = self.problem.col_tree
            far = self.problem.col_far
            far2 = self.problem.row_far
            transfer = self.col_transfer
            interaction = self.col_interaction
            interaction2 = self.row_interaction
            node_count = tree.level[-1]
        diff = [None for i in range(node_count)]
        level = tree.level
        level_count = len(level)-2
        for i in range(level_count-1, -1, -1):
            for j in range(level[i], level[i+1]):
                if transfer[j] is None:
                    continue
                # Update `transfer` matrix from children nodes
                if tree.child[j]:
                    s = 0
                    for l in tree.child[j]:
                        e = s+diff[l].shape[1]
                        transfer[j][s:e] = diff[l].dot(transfer[j][s:e])
                        s = e
                # Update `transfer` matrix with Q factor of QR factorization
                transfer[j], r = np.linalg.qr(transfer[j])
                # Apply R factor of QR factorization to `interaction` matrices
                for l in range(len(interaction[j])):
                    interaction[j][l] = np.tensordot(r, interaction[j][l], 1)
                diff[j] = r
        if self.problem.symmetric:
            for i in range(level_count-1, -1, -1):
                for j in range(level[i], level[i+1]):
                    # Apply R factors to `interaction` matrices from the
                    # other side for symmetry
                    for l in range(len(interaction[j])):
                        interaction[j][l] = np.tensordot(interaction[j][l],
                                diff[far[j][l]].T, 1)
        else:
            for i in range(level[-1]):
                # Update `interaction` matrices for another tree
                for j in range(len(interaction[i])):
                    l = far[i][j]
                    m = far2[l].index(i)
                    interaction2[l][m] = interaction[i][j].T

    def _compress(self, RC, tau):
        """
        Performs SVD decompression of block rows and columns.

        Parameters `RC` shows which cluster tree needs to be
        decompressed. Requires all transfer matrices of adjunct cluster
        tree to be orthogonal.

        Notes
        -----
        Works only in sequential programs.
        """
        if RC == 'row':
            tree = self.problem.row_tree
            far = self.problem.row_far
            far2 = self.problem.col_far
            basis_size = self.row_basis_size
            transfer = self.row_transfer
            interaction = self.row_interaction
            interaction2 = self.col_interaction
            node_count = tree.level[-1]
            notransition = self.problem.row_notransition
        else:
            tree = self.problem.col_tree
            far = self.problem.col_far
            far2 = self.problem.row_far
            basis_size = self.col_basis_size
            transfer = self.col_transfer
            interaction = self.col_interaction
            interaction2 = self.row_interaction
            node_count = tree.level[-1]
            notransition = self.problem.col_notransition
        diff = [None for i in range(node_count)]
        level = tree.level
        level_count = len(level)-2
        # Orthogonolize [all M2L matrices, part of parent M2M] of each node.
        # After this loop only `transfer` matrices hold singular values of
        # block rows/columns.
        for i in range(level_count):
            for k in range(level[i], level[i+1]):
                if notransition[k]:
                    continue
                # Put part of parent `transfer` matrix, according to
                # node itself, into 'tmp_matrix'
                # Put all the `interaction` matrices into 'tmp_matrix'
                tmp_list = []
                if interaction[k]:
                    tmp_matrix = np.concatenate([I.reshape(I.shape[0], -1) for
                        I in interaction[k]], axis=-1)
                    tmp_list.append(tmp_matrix)
                p = tree.parent[k]
                if p != -1 and transfer[p] is not None:
                    l = 0
                    child = tree.child[p]
                    ind0 = 0
                    ind1 = transfer[child[l]].shape[1]
                    while child[l] != k:
                        l += 1
                        ind0 = ind1
                        ind1 = ind1+transfer[child[l]].shape[1]
                    tmp_list.append(transfer[p][ind0:ind1])
                matrix = np.concatenate(tmp_list, axis=-1).T
                # Compute QR of `matrix`
                Q, R = np.linalg.qr(matrix)
                # Update self and parent `transfer` matrices
                transfer[k] = transfer[k].dot(R.T)
                basis_size[k] = R.shape[0]
                if matrix.shape[0] >= matrix.shape[1]:
                    if transfer[p] is not None:
                        transfer[p][ind0:ind1] = Q.T[:, -transfer[p].shape[1]:]
                else:
                    if transfer[p] is not None:
                        transfer[p] = np.concatenate([transfer[p][:ind0],
                            Q.T[:, -transfer[p].shape[1]:],
                            transfer[p][ind1:]], axis=0)
                # Update `interaction` matrices
                ind0 = 0
                for l in range(len(interaction[k])):
                    tmp_shape = np.array(interaction[k][l].shape)
                    ind1 = ind0+tmp_shape[1:].prod()
                    tmp_shape[0] = -1
                    interaction[k][l] = Q.T[:, ind0:ind1].reshape(tmp_shape)
                    ind0 = ind1
                if self.problem.symmetric:
                    for l in range(len(far[k])):
                        j = far[k][l]
                        m = far2[j].index(k)
                        interaction[j][m] = interaction[k][l].T
        # Truncate low singular values from `transfer` matrices
        for i in range(level_count-1, -1, -1):
            for k in range(level[i], level[i+1]):
                if transfer[k] is None:
                    continue
                # Restore `transfer` matrix (apply `diff` from children)
                child = tree.child[k]
                if child:
                    tmp_list = []
                    ind0 = 0
                    for l in child:
                        ind1 = ind0+diff[l].shape[1]
                        tmp_list.append(diff[l].dot(transfer[k][ind0:ind1]))
                        ind0 = ind1
                    transfer[k] = np.concatenate(tmp_list, axis=0)
                # SVD of block row/column via SVD of `transfer` matrix
                U, S, V = np.linalg.svd(transfer[k], full_matrices=0)
                new_rank = S.size
                tmp_eps = tau*S[0]
                # Acquire new rank
                for l in range(S.size):
                    if S[l] < tmp_eps:
                        new_rank = l
                        break
                # Cut low singular values
                S = S[:new_rank].copy()
                U = U[:, :new_rank].copy()
                V = V[:new_rank].copy()
                diff[k] = np.diag(S).dot(V)
                # Update `transfer` and `interaction` matrices
                transfer[k] = U
                basis_size[k] = new_rank
                for l in range(len(interaction[k])):
                    interaction[k][l] = np.tensordot(diff[k],
                            interaction[k][l], 1)
                if self.problem.symmetric:
                    for l in range(len(far[k])):
                        j = far[k][l]
                        m = far2[j].index(k)
                        interaction[j][m] = np.tensordot(interaction[j][m],
                                diff[k].T, 1)
        # Update interaction matrices for another tree
        if not self.problem.symmetric:
            for k in range(level[-1]):
                for l in range(len(interaction[k])):
                    j = far[k][l]
                    m = far2[j].index(k)
                    interaction2[j][m] = interaction[k][l].T

    def mcbh(self, onfly=None, verbose=False):
        """
        Converts current H2matrix representation to "mcbh" H2-type.

        Parameters
        ----------
        onfly : boolean
            If true, converts `mem_type` to "low", otherwise converts
            it to "full".
        verbose : boolean
            If true, outputs some additional information.

        Notes
        -----
        Works only in sequential programs.
        """
        if self.mpi_comm is not None:
            raise NotImplementedError("mcbh method works only in sequential"
                    " mode")
        t0 = time()
        if self.h2_type == 'mcbh' and self.mem_type == 'low'\
                and (onfly is None or onfly):
            if verbose:
                print('Already on required representation')
            return
        if self.h2_type == 'mcbh' and self.mem_type == 'full'\
                and (onfly is None or not onfly):
            if verbose:
                print('Already on required representation')
            return
        if self.h2_type == 'mcbh' and self.mem_type == 'low':
            self._matrix()
            self._close_matrix()
            self.mem_type = 'full'
            if verbose:
                print('Computed far-field and near-field interaction matrices'
                        'in {} seconds'.format(time()-t0))
            return
        if self.h2_type == 'mcbh' and self.mem_type == 'full':
            self.row_interaction = None
            self.col_interaction = None
            self.row_close = None
            self.col_close = None
            self.mem_type = 'low'
            if verbose:
                print('Removed interaction matrices')
            return
        tol = 1.05
        symmetric = self.problem.symmetric
        row = self.problem.row_tree
        col = self.problem.col_tree
        level_count = len(row.level)-2
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_far = self.problem.row_far
        col_far = self.problem.col_far
        row_basis = self.row_basis
        col_basis = self.col_basis
        row_transfer = self.row_transfer
        col_transfer = self.col_transfer
        row_notransition = self.problem.row_notransition
        col_notransition = self.problem.col_notransition
        row_basis = [None for i in range(row_size)]
        if not symmetric:
            col_basis = [None for i in range(col_size)]
        else:
            col_basis = row_basis
        self.row_basis = row_basis
        self.col_basis = col_basis
        row_r = [0 for i in range(row_size)]
        col_r = [0 for i in range(col_size)]
        for i in range(level_count-1, -1, -1):
            for j in range(col.level[i], col.level[i+1]):
                if col_notransition[j]:
                    continue
                if col.child[j] == []:
                    if col_transfer[j].shape[0] == 0:
                        col_transfer[j] = np.eye(col_transfer[j].shape[1])
                    tmp = maxvol_qr(col_transfer[j], tol)
                    col_r[j] = col_transfer[j][tmp[0]]
                    col_transfer[j] = tmp[1]
                    col_basis[j] = col.index[j][tmp[0]]
                else:
                    s = []
                    s2 = []
                    ind = 0
                    if col_transfer[j].shape[0] == 0:
                        col_transfer[j] = np.eye(col_transfer[j].shape[1])
                    for k in col.child[j]:
                        p = col_r[k].shape[0]
                        s.append(col_r[k].dot(col_transfer[j][ind:ind+p]))
                        s2.append(col_basis[k])
                        ind += p
                    tmp = maxvol_qr(np.concatenate(s, axis=0), tol)
                    col_r[j] = np.concatenate(s, axis=0)[tmp[0]]
                    col_transfer[j] = tmp[1]
                    col_basis[j] = np.concatenate(s2)[tmp[0]]
            if i < level_count-1:
                for j in range(col.level[i+1], col.level[i+2]):
                    col_r[j] = 0
        for i in range(col.level[-3], col.level[-1]):
            col_r[i] = 0
        if not symmetric:
            for i in range(level_count-1, -1, -1):
                for j in range(row.level[i], row.level[i+1]):
                    if row_notransition[j]:
                        continue
                    if row.child[j] == []:
                        if row_transfer[j].shape[0] == 0:
                            row_transfer[j] = np.eye(row_transfer[j].shape[1])
                        tmp = maxvol_qr(row_transfer[j], tol)
                        row_r[j] = row_transfer[j][tmp[0]]
                        row_transfer[j] = tmp[1]
                        row_basis[j] = row.index[j][tmp[0]]
                    else:
                        s = []
                        s2 = []
                        ind = 0
                        if row_transfer[j].shape[0] == 0:
                            row_transfer[j] = np.eye(row_transfer[j].shape[1])
                        for k in row.child[j]:
                            p = row_r[k].shape[0]
                            s.append(row_r[k].dot(row_transfer[j][ind:ind+p]))
                            s2.append(row_basis[k])
                            ind += p
                        tmp = maxvol_qr(np.concatenate(s, axis=0), tol)
                        row_r[j] = np.concatenate(s, axis=0)[tmp[0]]
                        row_transfer[j] = tmp[1]
                        row_basis[j] = np.concatenate(s2)[tmp[0]]
                if i < level_count-1:
                    for j in range(row.level[i+1], row.level[i+2]):
                        row_r[j] = 0
            for i in range(row.level[-3], row.level[-1]):
                row_r[i] = 0
        self.h2_type = 'mcbh'
        if onfly is None:
            if self.mem_type == 'low':
                self.row_interaction = None
                self.col_interaction = None
            else:
                self._matrix()
                self._close_matrix()
        elif not onfly:
            self._matrix()
            self._close_matrix()
        else:
            self.row_interaction = None
            self.col_interaction = None
            self.row_close = None
            self.col_close = None
        if verbose:
            print('Converted in {} seconds'.format(time()-t0))

    def copy(self, verbose=False):
        """
        Copies all buffers into new instance of H2matrix.

        Parameters
        ----------
        verbose : boolean
            If true, shows copy time.
        """
        symmetric = self.problem.symmetric
        t0 = time()
        far = self.problem.row_far
        close = self.problem.row_close
        size = self.problem.row_tree.level[-1]
        if symmetric:
            src = self.row_interaction
            if isinstance(src, list):
                dst = [[None for j in i] for i in far]
                for i in range(size):
                    for j in range(len(far[i])):
                        k = far[i][j]
                        if i <= k and src[i][j] is not None:
                            dst[i][j] = src[i][j].copy()
                            l = far[k].index(i)
                            dst[k][l] = dst[i][j].T
            elif isinstance(src, dict):
                dst = {}
                for i, src2 in src.iteritems():
                    for j, matrix in src2.iteritems():
                        if i <= j:
                            if i not in dst:
                                dst[i] = {}
                            dst[i][j] = matrix.copy()
                            if j not in dst:
                                dst[j] = {}
                            dst[j][i] = dst[i][j].T
            elif src is None:
                dst = None
            else:
                raise TypeError("row_interaction must be list/dict.")
            row_interaction = dst
            col_interaction = dst
            src = self.row_close
            if isinstance(src, list):
                dst = [[None for j in i] for i in close]
                for i in range(size):
                    for j in range(len(close[i])):
                        k = close[i][j]
                        if src[i][j] is not None:
                            if i <= k:
                                dst[i][j] = src[i][j].copy()
                            if i < k:
                                l = close[k].index(i)
                                dst[k][l] = dst[i][j].T
            elif isinstance(src, dict):
                dst = {}
                for i, src2 in src.iteritems():
                    for j, matrix in src2.iteritems():
                        if i <= j:
                            if i not in dst:
                                dst[i] = {}
                            dst[i][j] = matrix.copy()
                        if i < j:
                            if j not in dst:
                                dst[j] = {}
                            dst[j][i] = dst[i][j].T
            elif src is None:
                dst = None
            else:
                raise TypeError("row_close must be list/dict.")
            row_close = dst
            col_close = dst
            row_transfer = copy.deepcopy(self.row_transfer)
            col_transfer = row_transfer
            row_basis_size = self.row_basis_size.copy()
            col_basis_size = row_basis_size
            row_basis = copy.deepcopy(self.row_basis)
            col_basis = row_basis
        else:
            src = copy.deepcopy(self.col_interaction)
            far2 = self.problem.col_far
            if isinstance(src, list):
                dst = [[None for j in i] for i in far]
                for i in range(size):
                    for j in range(len(far[i])):
                        k = far[i][j]
                        l = far2[k].index(i)
                        if src[k][l] is not None:
                            dst[i][j] = src[k][l].T
            elif isinstance(src, dict):
                dst = {}
                for i, src2 in src.iteritems():
                    for j, matrix in src2.iteritems():
                        if j not in dst:
                            dst[j] = {}
                        dst[j][i] = matrix.T
            elif src is None:
                dst = None
            else:
                raise TypeError("col_interaction must be list/dict.")
            row_interaction = dst
            col_interaction = src
            src = copy.deepcopy(self.col_close)
            close2 = self.problem.col_close
            if isinstance(src, list):
                dst = [[None for j in i] for i in close]
                for i in range(size):
                    for j in range(len(close[i])):
                        k = close[i][j]
                        l = close2[k].index(i)
                        if src[k][l] is not None:
                            dst[i][j] = src[k][l].T
            elif isinstance(src, dict):
                dst = {}
                for i, src2 in src.iteritems():
                    for j, matrix in src2.iteritems():
                        if j not in dst:
                            dst[j] = {}
                        dst[j][i] = matrix.T
            elif src is None:
                dst = None
            else:
                raise TypeError("col_close must be list/dict.")
            row_close = dst
            col_close = src
            row_transfer = copy.deepcopy(self.row_transfer)
            col_transfer = copy.deepcopy(self.col_transfer)
            row_basis_size = self.row_basis_size.copy()
            col_basis_size = self.col_basis_size.copy()
            row_basis = copy.deepcopy(self.row_basis)
            col_basis = copy.deepcopy(self.col_basis)
        ans = H2matrix(self.problem, row_basis_size, col_basis_size,
                row_transfer, col_transfer, row_interaction, col_interaction,
                row_close, col_close, row_basis, col_basis, self.mpi_comm)
        if verbose:
            print('Copied in {} seconds'.format(time()-t0))
        return ans

    def diffnorm(self, factor2=None, far_only=False):
        """
        Computes relative spectral distance from H2matrix.

        If `factor2` is not given, distance is measured to initial
        linear operator, represented by `problem` attribute. Otherwise,
        distance is measured to `factor2` operator with help of
        following methods, defined in `factor2` object: `dot`, `rdot`,
        `far_dot` and `far_rdot`. Meaning of this methods is the same,
        as meaning of methods of this object.

        If parameter `far_only` is True, measures relative error of
        approximation of far-field part only.

        Parameters
        ----------
        factor2 : Python object
            If not defined, this function returns relative spectral
            error of approximation by H2matrix. If defined, this
            method returns relative spectral distance from H2matrix
            to `factor2`
        far_only : boolean
            If true, measures distance only by far-field part.

        Returns
        -------
        float
            Relative spectral distance to initial operator or `factor2`
            object.
        """
        try:
            svdp
        except NameError:
            raise ImportError("No pypropack installed, cannot measure error.")
        if factor2 is None:
            factor2 = self.problem
        # Fix to have the same random vector on all MPI nodes
        np.random.seed(int(time()))
        if far_only:
            linop_diff = la.LinearOperator(self.shape, matvec=lambda x:
                    factor2.far_dot(x)-self.far_dot(x), rmatvec=lambda x:
                    factor2.far_rdot(x)-self.far_rdot(x), dtype=self.dtype)
            s_diff = svdp(linop_diff, 1, compute_u=0, compute_v=0, kmax=100)
            linop_factor = la.LinearOperator(self.shape, matvec=self.far_dot,
                    rmatvec=self.far_rdot, dtype=self.dtype)
            s_factor = svdp(linop_factor, 1, compute_u=0, compute_v=0,
                    kmax=100)
        else:
            linop_diff = la.LinearOperator(self.shape, matvec=lambda x:
                        factor2.dot(x)-self.dot(x), rmatvec=lambda x:
                        factor2.rdot(x)-self.rdot(x), dtype=self.dtype)
            s_diff = svdp(linop_diff, 1, compute_u=0, compute_v=0, kmax=100)
            linop_factor = la.LinearOperator(self.shape, matvec=lambda x:
                    self.dot(x), rmatvec=lambda x:self.rdot(x),
                    dtype=self.dtype)
            s_factor = svdp(linop_factor, 1, compute_u=0, compute_v=0,
                    kmax=100)
        return s_diff[0][0]/s_factor[0][0]

    def __str__(self):
        mpi_comm = self.mpi_comm
        old_printoptions = np.get_printoptions()
        np.set_printoptions(precision=2, suppress=1)
        if mpi_comm is not None:
            mpi_rank = mpi_comm.rank
            mpi_size = mpi_comm.size
            mem = np.zeros((mpi_size, 5), np.float)
            send_buf = np.zeros(5, np.float)
            send_buf[0] = self.nbytes(0, 0, 0, 1)
            send_buf[1] = self.nbytes(1, 0, 0, 0)
            send_buf[2] = self.nbytes(0, 1, 0, 0)
            send_buf[3] = self.nbytes(0, 0, 1, 0)
            send_buf[4] = self.nbytes()
            send_buf /= 2**20
            mpi_comm.Allgather(send_buf, mem)
            mem = mem.T
            out = ('H2matrix on {} MPI nodes at {}\n'+
                    '    Structure (h2/mcbh): {}\n'+
                    '    Memory layout (low/full): {}\n'+
                    '    Shape: {}\n'+
                    '    Total memory, MB: {}\n'+
                    '        Basises, MB: {}\n'+
                    '        Transfer matrices, MB: {}\n'+
                    '        Far-field interactions, MB: {}\n'+
                    '        Near-field interactions, MB: {}')
            out = out.format(mpi_size, hex(id(self)), self.h2_type,
                    self.mem_type, [self.shape[0]]+
                    list(self.element_shape)+[self.shape[1]], mem[4], mem[0],
                    mem[1], mem[2], mem[3])
        else:
            mem = np.zeros(5, np.float)
            mem[0] = self.nbytes(0, 0, 0, 1)
            mem[1] = self.nbytes(1, 0, 0, 0)
            mem[2] = self.nbytes(0, 1, 0, 0)
            mem[3] = self.nbytes(0, 0, 1, 0)
            mem[4] = self.nbytes()
            mem /= 2**20
            out = ('H2matrix at {}\n'+
                    '    Structure (h2/mcbh): {}\n'+
                    '    Memory layout (low/full): {}\n'+
                    '    Shape: {}\n'+
                    '    Total memory, MB: {:.2f}\n'+
                    '        Basises, MB: {:.2f}\n'+
                    '        Transfer matrices, MB: {:.2f}\n'+
                    '        Far-field interactions, MB: {:.2f}\n'+
                    '        Near-field interactions, MB: {:.2f}')
            out = out.format(hex(id(self)), self.h2_type, self.mem_type,
                    [self.shape[0]]+list(self.element_shape)+[self.shape[1]],
                    mem[4], mem[0], mem[1], mem[2], mem[3])
        np.set_printoptions(**old_printoptions)
        return out
