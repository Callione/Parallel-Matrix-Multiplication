import random
from tqdm import tqdm
import numpy as np

class Matrix:
    def __init__(self,row,col=None,use_np=False):
        self.use_np = use_np
        if col:
            self.shape = (row,col)
        else:
            n = row
            self.shape = (n,n)
        self.matrix = [[0]*self.shape[1] for _ in range(self.shape[0])]
        
    def set_value(self,low=0,high=10):
        n = self.shape[0]
        for i in range(n):
            for j in range(n):
                self.matrix[i][j] = random.randint(low,high)
                
    def print(self):
        max_len = max(len(str(self.matrix[i][j])) for i in range(self.shape[0]) for j in range(self.shape[1]))
        row_format = "{:>" + str(max_len) + "}"

        for i in range(self.shape[0]):
            row_str = [row_format.format(self.matrix[i][j]) for j in range(self.shape[1])]
            print(" ".join(row_str))
                
    def multiply(self, other_matrix):
        if self.shape[1] != other_matrix.shape[0]:
            print("Invalid matrix dimensions for multiplication")
            return None

        result_matrix = Matrix(self.shape[0], other_matrix.shape[1],use_np=self.use_np)

        if self.use_np:
            A = np.array(self.matrix)
            B = np.array(other_matrix.matrix)
            Prod = np.dot(A,B)
            result_matrix.matrix = Prod.tolist()
            return result_matrix
        

        #for i in tqdm(range(self.shape[0]),desc='multiply'):
        for i in range(self.shape[0]):
            for j in range(other_matrix.shape[1]):
                for k in range(self.shape[1]):
                    result_matrix.matrix[i][j] += self.matrix[i][k] * other_matrix.matrix[k][j]

        return result_matrix
    
    
    def add(self,other_matrix):
        if self.shape != other_matrix.shape:
            print('A.add(B) requires the same shape')
            return None
        
        sum = Matrix(self.shape[0],self.shape[1],use_np=self.use_np)
        
        if self.use_np:
            A = np.array(self.matrix)
            B = np.array(other_matrix.matrix)
            sum.matrix = (A + B).tolist()
            return sum
        
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                sum.matrix[r][c] += self.matrix[r][c] + other_matrix.matrix[r][c]
                
        return sum
            
    
    def split_blocks(self,q):
        # matrix will be split into (q x q) blocks
        # p = q*q is the number of processes
        
        # only works for square matrix
        assert self.shape[0] == self.shape[1], "Only square matrix can be split into blocks"
        
        block_size = self.shape[0]//q
        blocks = []
        for i in range(q):
            for j in range(q):
                
                block_ij = Matrix(block_size,use_np=self.use_np)
                for row in range(block_size):
                    block_ij.matrix[row] = self.matrix[i*block_size+row][j*block_size:(j+1)*block_size]
                
                blocks.append(block_ij)
        
        return blocks