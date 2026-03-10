"""
Reusable MPI utilities for distributing ensemble sets across ranks.

Provides point-to-point communication patterns (send/recv) that avoid
MPI sub-communicators (comm.Split), which crash on some OpenMPI + libfabric
installations. All functions gracefully fall back to serial mode when
mpi4py is unavailable.
"""

# Conditional MPI import with serial fallback
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def get_comm():
    """Get MPI communicator, rank, and size.

    Returns (comm, rank, size). When MPI is unavailable or the script
    was not launched with mpirun, returns (None, 0, 1) for serial mode.
    """
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size == 1:
            return None, 0, 1
        return comm, rank, size
    return None, 0, 1


def get_set_assignments(rank, size, n_sets):
    """Assign ensemble sets to MPI ranks.

    Returns a list of (set_id, local_rank, local_size) tuples describing
    the work this rank should perform.

    Two regimes:
    - size >= n_sets: Each rank is assigned to exactly one set.
      Multiple ranks collaborate on the same set (local_size > 1).
    - size < n_sets: Each rank processes one or more sets serially,
      each with local_rank=0, local_size=1.
    """
    if size >= n_sets:
        ranks_per_set = size // n_sets
        set_id = rank // ranks_per_set
        # Fold leftover ranks into the last set
        if set_id >= n_sets:
            set_id = n_sets - 1
        local_rank = rank - set_id * ranks_per_set
        # Calculate actual size for this set (last set may have extras)
        if set_id < n_sets - 1:
            local_size = ranks_per_set
        else:
            local_size = size - set_id * ranks_per_set
        return [(set_id, local_rank, local_size)]
    else:
        # More sets than ranks: distribute sets round-robin
        assignments = []
        for s in range(n_sets):
            if s % size == rank:
                assignments.append((s, 0, 1))
        return assignments


def get_set_peer_ranks(rank, size, n_sets):
    """Return sorted list of global ranks assigned to the same set as `rank`.

    Only meaningful when size >= n_sets. Returns [rank] otherwise.
    """
    if size < n_sets:
        return [rank]

    ranks_per_set = size // n_sets
    set_id = rank // ranks_per_set
    if set_id >= n_sets:
        set_id = n_sets - 1

    peers = []
    for r in range(size):
        r_set = r // ranks_per_set
        if r_set >= n_sets:
            r_set = n_sets - 1
        if r_set == set_id:
            peers.append(r)
    return sorted(peers)


def point_to_point_gather(comm, data, local_rank, local_size, set_peer_ranks):
    """Gather data to local_rank 0 using point-to-point send/recv.

    Avoids collective operations on sub-communicators.

    Returns:
        On local_rank 0: list of data from all peers (ordered by local_rank).
        On other ranks: None.
    """
    if local_size <= 1:
        return [data]

    root_global = set_peer_ranks[0]

    if local_rank == 0:
        gathered = [None] * local_size
        gathered[0] = data
        for lr in range(1, local_size):
            sender_global = set_peer_ranks[lr]
            gathered[lr] = comm.recv(source=sender_global, tag=200 + lr)
        return gathered
    else:
        comm.send(data, dest=root_global, tag=200 + local_rank)
        return None


def point_to_point_bcast(comm, data, local_rank, set_peer_ranks):
    """Broadcast data from local_rank 0 using point-to-point send/recv.

    Returns the broadcast data on all ranks.
    """
    local_size = len(set_peer_ranks)
    if local_size <= 1:
        return data

    root_global = set_peer_ranks[0]

    if local_rank == 0:
        for lr in range(1, local_size):
            dest_global = set_peer_ranks[lr]
            comm.send(data, dest=dest_global, tag=300 + lr)
        return data
    else:
        data = comm.recv(source=root_global, tag=300 + local_rank)
        return data


def point_to_point_barrier(comm, local_rank, local_size, set_peer_ranks):
    """Synchronize set peers using point-to-point send/recv.

    All ranks send a signal to local_rank 0, which then replies,
    ensuring everyone has reached this point before continuing.
    """
    if local_size <= 1:
        return

    root_global = set_peer_ranks[0]

    if local_rank == 0:
        # Receive ready signal from all peers
        for lr in range(1, local_size):
            sender_global = set_peer_ranks[lr]
            comm.recv(source=sender_global, tag=400 + lr)
        # Send go-ahead to all peers
        for lr in range(1, local_size):
            dest_global = set_peer_ranks[lr]
            comm.send(True, dest=dest_global, tag=500 + lr)
    else:
        comm.send(True, dest=root_global, tag=400 + local_rank)
        comm.recv(source=root_global, tag=500 + local_rank)


def global_point_to_point_gather(comm, data, rank, size, tag=600):
    """Gather data from all ranks to rank 0 using point-to-point send/recv.

    Replacement for comm.gather() that avoids collective operations on
    COMM_WORLD, which crash on some OpenMPI + libfabric installations.

    Returns:
        On rank 0: list of data from all ranks (ordered by rank).
        On other ranks: None.
    """
    if comm is None or size <= 1:
        return [data]

    if rank == 0:
        gathered = [None] * size
        gathered[0] = data
        for r in range(1, size):
            gathered[r] = comm.recv(source=r, tag=tag)
        return gathered
    else:
        comm.send(data, dest=0, tag=tag)
        return None
