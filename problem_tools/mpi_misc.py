from __future__ import print_function, division, absolute_import

try:
    from mpi4py import MPI
except:
    pass
import numpy as np
from sys import getsizeof

def sync_mpi_data(s_data, r_data, mpi_comm):
    """
    Synchronization of data and corresponding index, given as a dict.

    Calls MPI alltoall procedure to scatter data, based on index % MPI
    size, then sums up all data with equal index and, finally, calls
    MPI allgather to synchronize results.

    Parameters
    ----------
    s_data : dict
        Data, represented by a dictionary.
    r_data : list
        Place, where to hold result.
    mpi_comm : MPI communicator.
    """
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
    send_buffer = [{} for i in range(mpi_size)]
    for ind, data in s_data.iteritems():
        cur_dict = send_buffer[ind % mpi_size]
        if ind in cur_dict:
            cur_dict[ind] += s_data[ind]
        else:
            cur_dict[ind] = s_data[ind]
    tmp_data = mpi_comm.alltoall(send_buffer)
    cur_dict = tmp_data[0]
    for i in range(1, mpi_size):
        for ind, data in tmp_data[i].iteritems():
            if ind in cur_dict:
                cur_dict[ind] += data
            else:
                cur_dict[ind] = data
    tmp_data = mpi_comm.allgather(cur_dict)
    for cur_dict in tmp_data:
        for ind, data in cur_dict.iteritems():
            r_data[ind] = data
