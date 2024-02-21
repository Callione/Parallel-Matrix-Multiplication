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
        
        # # see the efficiency of numpy first
        
        # A_np = np.array(A.matrix)
        # B_np = np.array(B.matrix)
        
        # start = time.time()
        # C_np = np.dot(A_np,B_np)
        # end   = time.time()
        # np_time = end - start
        # print(f'numpy:\t{np_time}s')
        
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
        
        #------- circular shift of A_blocks
        A_blocks_shift = [None]*num_proc
        for r in range(q):
            for c in range(q):
                c_shift = (c - r + q)%q
                A_blocks_shift[r*q + c_shift] = A_blocks[r*q+c]
        #------- circular shift of B_blocks
        B_blocks_shift = [None]*num_proc
        for c in range(q):
            for r in range(q):
                r_shift = (r - c + q)%q
                B_blocks_shift[r_shift*q + c] = B_blocks[r*q+c]  
                            
                    
        for i in range(q):
            for j in range(q):
                scatter_buf.append((A_blocks_shift[i*q+j],B_blocks_shift[i*q+j]))
    else:
        scatter_buf = None

    scatter_start = time.time()
    # -------scatter local block (Aij,Bij) --------    
    recv_buf = comm.scatter(scatter_buf,root=0)
    scatter_end = time.time()
    print(f'rank{rank}')
    scatter_time = scatter_end-scatter_start
    print(f'scatter block {scatter_time}s')
    
    # get local block Aij Bij
    A_ij,B_ij = recv_buf
    
    A_ik,B_kj = A_ij,B_ij            
    Cij = Matrix(block_size,use_np=args.np)
    
    row = rank //q
    col = rank % q
    
    mm_start = time.time()
    for k in range(q):
        Cij = Cij.add(A_ik.multiply(B_kj))

        if k == q-1:
            break
        
        # send current A_ik to the left process and receive new A_ij from the right
        dest_rank = row*q + (col - 1 + q)%q
        src_rank  = row*q + (col +1)%q
        
        if(col == 0):
            new_A_ik = comm.recv(source=src_rank)
            comm.send(A_ik, dest_rank)
            A_ik = new_A_ik
        else:
            comm.send(A_ik, dest_rank)
            A_ik = comm.recv(source=src_rank)        
        # update A_ik with newly received block
        
        
        
        # send current B_kj to the upper process and receive new B_ij from below
        dest_rank = ((row -1 +q)%q)*q + col
        src_rank  = ((row+1)%q)*q + col
        

        if(row == 0):
            new_B_kj =comm.recv(source=src_rank)
            comm.send(B_kj,dest_rank)
            B_kj = new_B_kj

        else:
            comm.send(B_kj,dest_rank)
            B_kj =comm.recv(source=src_rank)            
        # update B_kj with newly received block
        
    mm_end = time.time()
    print(f'local multiply:{mm_end-mm_start}s')
    
    
    # Gather Cij from all processes
    C_blocks = comm.gather(Cij.matrix, root=0)
    end = time.time()
    parallel_time = end - start
    time_cost = comm.gather(parallel_time,root=0)
    print(f'rank{rank},parallel:\t{end-start}s')
    print(f'scatter:\t{scatter_time/parallel_time*100:.3f}%')
    

    
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
        

