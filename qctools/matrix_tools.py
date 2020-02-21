# coding:utf-8
import numpy as np
from scipy.ndimage.interpolation import shift

class QCMatrix(object):
    '''
    Attributes
    ==========
    nrow : int
        number of rows in QC-matrix
    ncol : int
        number of columns in QC-matrix
    gamma : int (>=1)
        hyperparameter for QC-matrix
    '''

    def __init__(self, nrow=None, ncol=None, gamma=4):
        self.nrow = nrow
        self.ncol = ncol
        self.gamma = gamma

    def _make_shifttable(self, nrow, ncol, gamma, col_shift=0, row_shift=0):
        '''
        make a shift table for each matrix (size: `nrow` x `ncol`) from fixed hyperparameter `gamma`.
        '''
        tbl = np.empty((nrow, ncol), dtype=np.int16)
        for i in range(nrow):
            for j in range(ncol):
                tbl[i, j] = (i + row_shift) * (j + col_shift) % gamma
        return tbl

    def build(self, nrow, ncol):
        '''
        make QC-Matrix (size: `nrow` x `ncol`) from fixed hyperparameter `gamma`.
        '''
        # set arguments
        if self.nrow is None: self.nrow = nrow
        if self.ncol is None: self.ncol = ncol

        # QC-Matrix
        qc_mat = np.zeros((self.nrow, self.ncol), dtype=np.int16)  
        # Size of each block matrix in QC-Matrix
        block_row_nb = int(np.ceil(self.nrow / self.gamma) ) 
        block_col_nb = int(np.ceil(self.ncol / self.gamma) )
        # Shit-Table
        shift_table = self._make_shifttable(block_row_nb, block_col_nb, self.gamma)
        # Identity matrix
        base_block = np.eye(int(self.gamma), dtype=np.int16)

        for i in range(block_row_nb):
            for j in range(block_col_nb):

                # s: shift size
                s = shift_table[i, j]
                # block: block matrix in QC-Matrix
                block = np.roll(base_block, s, axis=0)

                for x in range(self.gamma):
                    for y in range(self.gamma):
                        row = i * self.gamma + x
                        col  = j * self.gamma + y

                        if row >= self.nrow or col >= self.ncol:
                            break
                        qc_mat[row, col] = block[x, y]

        return qc_mat

class QCMatrixWithSpatialCoupling(object):
    '''
    Attributes
    ==========
    nrow : int
        number of rows in QC-matrix
    ncol : int
        number of columns in QC-matrix
    gamma : int (>=1)
        hyperparameter for QC-matrix
    col_weight : int (>=3)
        hyperparameter for SpatialCoupling
    '''

    def __init__(self, nrow=None, ncol=None, gamma=4, col_weight=3):
        self.nrow = nrow
        self.ncol = ncol
        self.gamma = gamma
        self.col_weight = col_weight

    def _make_shifttable(self, nrow, ncol, gamma, col_shift=0, row_shift=0):
        '''
        make a shift table for each matrix (size: `nrow` x `ncol`) from fixed hyperparameter `gamma`.
        '''
        tbl = np.empty((nrow, ncol), dtype=np.int16)
        for i in range(nrow):
            for j in range(ncol):
                tbl[i, j] = (i + row_shift) * (j + col_shift) % gamma
        return tbl

    def _spatialcoupling_col(self, block_nrow, block_ncol, col_weight):
        '''
        make the binary matrix which supports Spatial-Coupling method when `nrow < ncol`.
        '''
        # binary matrix for spatial coupling
        sc_mat = np.zeros((block_nrow, block_ncol), dtype=np.int16)  

        for i in range(block_nrow):
            for j in range(block_ncol):
                # Mask Condition
                division = block_nrow - col_weight + 1
                diff = i - (j % division)
                if (0 <= diff) and (diff < col_weight) and (col_weight - diff - 1 + i < block_nrow): # 
                    sc_mat[i, j] = 1
        return sc_mat

    def _spatialcoupling_row(self, block_nrow, block_ncol, col_weight=3):
            '''
            make binary matrix which supports Spatial-Coupling method when nrow > ncol
            '''
            # binary matrix for spatial coupling
            sc_mat = np.zeros((block_nrow, block_ncol), dtype=np.int16)

            u = block_nrow
            d = block_nrow
            for di in range(block_nrow, 1, -1):
                u_new = int((block_nrow - di * (block_ncol - 1)))
                if col_weight <= u_new and u_new < u:
                    u = u_new
                    d = di
            if u == block_nrow and d == block_nrow:
                raise ValueError("There is no solusion which satisfies the constraint of arguments: col_weight="+str(col_weight))

            for i in range(block_nrow):
                for j in range(block_ncol):
                    # Mask Condition
                    if (d * j <= i) and (i < d * j + u):
                        sc_mat[i, j] = 1
            return sc_mat

    def build(self, nrow, ncol):
        '''
        make QC-Matrix (size: `nrow` x `ncol`) from fixed hyperparameter `gamma`.
        '''
        # set arguments
        if self.nrow is None: self.nrow = nrow
        if self.ncol is None: self.ncol = ncol

        # QC-Matrix
        qc_sc_mat = np.zeros((self.nrow, self.ncol), dtype=np.int16)  
        # Size of each block matrix in QC-Matrix
        block_row_nb = int(np.ceil(self.nrow / self.gamma) ) 
        block_col_nb = int(np.ceil(self.ncol / self.gamma) )
        # Shit-Table
        shift_table = self._make_shifttable(block_row_nb, block_col_nb, self.gamma)
        # Identity matrix
        base_block = np.eye(int(self.gamma), dtype=np.int16)

        # Spatial Coupling matrix
        if nrow > ncol:
            sc_mat = self._spatialcoupling_row(block_row_nb, block_col_nb, self.col_weight)
        elif nrow <= ncol:
            sc_mat = self._spatialcoupling_col(block_row_nb, block_col_nb, self.col_weight)

        for i in range(block_row_nb):
            for j in range(block_col_nb):

                if sc_mat[i, j] == 1:
                    
                    # s: shift size
                    s = shift_table[i, j]
                    # block: block matrix in QC-Matrix
                    block = np.roll(base_block, s, axis=0)

                    for x in range(self.gamma):
                        for y in range(self.gamma):
                            row = i * self.gamma + x
                            col  = j * self.gamma + y

                            if row >= self.nrow or col >= self.ncol:
                                break
                            qc_sc_mat[row, col] = block[x, y]

        return qc_sc_mat
