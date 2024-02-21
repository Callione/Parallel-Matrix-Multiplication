from mpi4py import MPI
import numpy as np
import time
import random
import argparse
from Matrix import Matrix


        
        
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',type=int,default=8,help='size of the square matrix')
    parser.add_argument('--v',default=False,action='store_true',help='whether to print the matrix')
    parser.add_argument('--np',default=False,action='store_true',help='whether to use numpy for calculation')
    
    args = parser.parse_args()
    n = args.N
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_proc = comm.Get_size()
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
    
    bcast_start = time.time()
    # Broadcast A,B to all processes
    A = comm.bcast(A,root=0)
    B = comm.bcast(B,root=0)
    
    bcast_end = time.time()
    print(f'rank{rank}')
    bcast_time = bcast_end - bcast_start
    print(f'bcast time={bcast_time}s')
    
    # get process index (i,j)
    i,j = 0,0
    q = int(num_proc**0.5)
    i = rank //q
    j = rank % q
    block_size = n//q
    
    # calculate block C(i,j) = \Sum_{k=0}^{q-1} A(i,k)*B(k,j)
    Cij = Matrix(block_size,use_np=args.np)

    mul_start = time.time()
    
    # get local strip A_row and B_col   
    A_row_i = Matrix(block_size,n,use_np=args.np)
    A_row_i.matrix = A.matrix[i*block_size:(i+1)*block_size]
    B_col_j = Matrix(n,block_size,use_np=args.np)
    for r in range(n):
        B_col_j.matrix[r] = B.matrix[r][j*block_size:(j+1)*block_size]
                
    Cij = A_row_i.multiply(B_col_j)
    
    mul_end = time.time()
    print(f'local multiply:{mul_end-mul_start}s')
                
    
    # Gather Cij from all processes
    C_blocks = comm.gather(Cij.matrix, root=0)
    end = time.time()
    parallel_time = end - start
    print(f'rank{rank} parallel:\t{parallel_time}s')
    print(f'communication:\t{bcast_time/parallel_time*100:.3f}%')

    time_cost = comm.gather(parallel_time,root=0)

    
    # Reconstruct the final matrix on rank 0
    if rank == 0:
        C = Matrix(n,use_np=args.np)
        for i in range(q):
            for j in range(q):
                for row in range(i*block_size,(i+1)*block_size):
                    C.matrix[row][j*block_size:(j+1)*block_size] = C_blocks[i*q+j][row%block_size]
        
        if args.v:
            print('\nParallel C\n------------------')
            C.print()
        
        print(f'parallel:\t{max(time_cost)}s')


        start = time.time()
        seq_C = A.multiply(B)
        end = time.time()
        seq_time = end - start
        
        if args.v:
            print('\nSequential C\n------------------')
            seq_C.print()
        
        print(f'sequential:\t{seq_time}s')
        
        print(f'C equal:{C.matrix == seq_C.matrix}')
        print(f'acceleration ratio:{seq_time/max(time_cost)}')
