from mpi4py import MPI
import numpy as np
import time
import random
import argparse
from Matrix import Matrix

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',type=int,default=4,help='size of the square matrix')
    parser.add_argument('--v',default=False,action='store_true',help='whether to print the matrix')
    parser.add_argument('--np',default=False,action='store_true',help='whether to use numpy for calculation')
    
    args = parser.parse_args()
    n = args.N
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_proc = comm.Get_size()
    q = int(num_proc**0.5)
    block_size = n//q
    
    if rank == 0:
        # initialize A and B
        print(f'num_proc = {num_proc}')
        print(f'shape=({n},{n})')
        A = Matrix(n,use_np=args.np)
        A.set_value()
        B = Matrix(n,use_np=args.np)
        B.set_value()
        
        if args.v:
            print('\nMatrix A\n--------------------')
            A.print()
            print('\nMatrix B\n------------------')
            B.print()
            
    else:
        A = None
        B = None
    
    start = time.time()
    if rank == 0:
        
        #------- prepare blocks for scatter ------
        scatter_buf = []
        A_blocks = A.split_blocks(q)
        B_blocks = B.split_blocks(q)
        
        for i in range(q):
            for j in range(q):
                scatter_buf.append((A_blocks[i*q+j],B_blocks[i*q+j]))
    else:
        scatter_buf = None

    scatter_start = time.time()
    # -------scatter local block--------    
    recv_buf = comm.scatter(scatter_buf,root=0)
    scatter_end = time.time()
    scatter_time = scatter_end-scatter_start
    print(f'rank{rank}')
    print(f'scatter block={scatter_time}s')
    
    # get local block Aij Bij
    A_ij,B_ij = recv_buf
        
    
    exchange_start = time.time()
    # exchange Aij with other processes in the same row
    row_number = rank // q
    row_comm = comm.Split(row_number,rank)
    A_row_i = row_comm.alltoall([A_ij]*q)

    # exchange Bij with other processes in the same col
    col_number = rank % q
    col_comm = comm.Split(col_number,rank)
    B_col_j = col_comm.alltoall([B_ij]*q)
    exchange_end = time.time()
    exchange_time = exchange_end-exchange_start
    print(f'exchange block={exchange_time}s')
    
    
    
    Cij = Matrix(block_size)
    for k in range(q):
        Cij = Cij.add(A_row_i[k].multiply(B_col_j[k]))

    
    
    # Gather Cij from all processes
    C_blocks = comm.gather(Cij.matrix, root=0)
    end = time.time()
    parallel_time = end - start
    print(f'rank{rank},parallel:\t{parallel_time}s')
    print(f'scatter:\t{scatter_time/parallel_time*100:.3f}%')
    print(f'exchange:\t{exchange_time/parallel_time*100:.3f}%')

    time_cost = comm.gather(parallel_time,root=0)

    
    # Reconstruct the final matrix on rank 0
    if rank == 0:
        C = Matrix(n,use_np=args.np)
        for i in range(q):
            for j in range(q):
                for row in range(i*block_size,(i+1)*block_size):
                    C.matrix[row][j*block_size:(j+1)*block_size] = C_blocks[i*q+j][row%block_size]


        print(f'parallel:\t{max(time_cost)}s')

        start = time.time()
        seq_C = A.multiply(B)
        end = time.time()
        seq_time = end - start

        if args.v:
            print('\nParallel C\n------------------')
            C.print()
            print('\nSequential C\n------------------')
            seq_C.print()

    
        print(f'sequential:\t{seq_time}s')
        print(f'C equal:{C.matrix == seq_C.matrix}')
        print(f'acceleration ratio:{seq_time/max(time_cost)}')
        

