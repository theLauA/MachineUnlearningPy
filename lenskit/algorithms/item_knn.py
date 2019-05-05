"""
Item-based k-NN collaborative filtering.
"""

import pathlib
import logging
import warnings
import time

import pandas as pd
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit, prange
from itertools import combinations 
from lenskit import util, matrix, DataWarning
from . import Predictor

_logger = logging.getLogger(__name__)


@njit(nogil=True)
def _predict_weighted_average(model, nitems, nrange, ratings, targets):
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in prange(targets.shape[0]):
        iidx = targets[i]
        rptr = model.rowptrs[iidx]
        rend = model.rowptrs[iidx + 1]

        num = 0
        denom = 0
        nnbrs = 0

        for j in range(rptr, rend):
            nidx = model.colinds[j]
            if np.isnan(ratings[nidx]):
                continue

            nnbrs = nnbrs + 1
            num = num + ratings[nidx] * model.values[j]
            denom = denom + np.abs(model.values[j])

            if max_nbrs > 0 and nnbrs >= max_nbrs:
                break

        if nnbrs < min_nbrs:
            continue

        scores[iidx] = num / denom

    return scores


@njit(nogil=True)
def _predict_sum(model, nitems, nrange, ratings, targets):
    min_nbrs, max_nbrs = nrange
    scores = np.full(nitems, np.nan, dtype=np.float_)

    for i in prange(targets.shape[0]):
        iidx = targets[i]
        rptr = model.rowptrs[iidx]
        rend = model.rowptrs[iidx + 1]

        score = 0
        nnbrs = 0

        for j in range(rptr, rend):
            nidx = model.colinds[j]
            if np.isnan(ratings[nidx]):
                continue

            nnbrs = nnbrs + 1
            score = score + model.values[j]

            if max_nbrs > 0 and nnbrs >= max_nbrs:
                break

        if nnbrs < min_nbrs:
            continue

        scores[iidx] = score

    return scores


_predictors = {
    'weighted-average': _predict_weighted_average,
    'sum': _predict_sum
}


class ItemItem(Predictor):
    """
    Item-item nearest-neighbor collaborative filtering with ratings. This item-item implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.

    Attributes:
        item_index_(pandas.Index): the index of item IDs.
        item_means_(numpy.ndarray): the mean rating for each known item.
        item_counts_(numpy.ndarray): the number of saved neighbors for each item.
        sim_matrix_(matrix.CSR): the similarity matrix.
        user_index_(pandas.Index): the index of known user IDs for the rating matrix.
        rating_matrix_(matrix.CSR): the user-item rating matrix for looking up users' ratings.
    """

    def __init__(self, nnbrs, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        """
        Args:
            nnbrs(int):
                the maximum number of neighbors for scoring each item (``None`` for unlimited)
            min_nbrs(int): the minimum number of neighbors for scoring each item
            min_sim(double): minimum similarity threshold for considering a neighbor
            save_nbrs(double):
                the number of neighbors to save per item in the trained model
                (``None`` for unlimited)
            center(bool):
                whether to normalize (mean-center) rating vectors.  Turn this off when working
                with unary data and other data types that don't respond well to centering.
            aggregate:
                the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        """
        self.nnbrs = nnbrs
        if self.nnbrs is not None and self.nnbrs < 1:
            self.nnbrs = -1
        self.min_nbrs = min_nbrs
        if self.min_nbrs is not None and self.min_nbrs < 1:
            self.min_nbrs = 1
        self.min_sim = min_sim
        self.save_nbrs = save_nbrs
        self.center = center
        self.aggregate = aggregate
        try:
            self._predict_agg = _predictors[aggregate]
        except KeyError:
            raise ValueError('unknown aggregator {}'.format(aggregate))

    def fit(self, ratings):
        """
        Train a model.

        The model-training process depends on ``save_nbrs`` and ``min_sim``, but *not* on other
        algorithm parameters.

        Args:
            ratings(pandas.DataFrame):
                (user,item,rating) data for computing item similarities.
        """
        # Training proceeds in 2 steps:
        # 1. Normalize item vectors to be mean-centered and unit-normalized
        # 2. Compute similarities with pairwise dot products
        self._timer = util.Stopwatch()

        init_rmat, users, items = matrix.sparse_ratings(ratings)
        
        for index, row in ratings.iterrows():
            if items.get_loc(row['item']) in [17,138,22,83,76,31,92]:
                #print(row['user'],row['item'],index,users.get_loc(row['user']),items.get_loc(row['item']))
                pass
        n_items = len(items)
        _logger.info('[%s] made sparse matrix for %d items (%d ratings from %d users)',
                     self._timer, len(items), init_rmat.nnz, len(users))

        start = time.time()
        rmat_scipy = init_rmat.to_scipy()
        #self._compute_similarities_unlearn(ratings,init_rmat,items,users)
        #self._compute_similarities_unlearn_min_centering(ratings,init_rmat,items,users)
        #self._unlearn_min_centering(54,17,init_rmat,self.smat_unlearn)
        
        self._compute_similarities_unlearn_min_centering_sparse_vectorize(rmat_scipy,items,users)
        #self._compute_similarities_unlearn_min_centering_matrix_vectorize(rmat_scipy,items,users)
        #self._compute_similarities_unlearn_global_centering_matrix_vectorize(rmat_scipy,items,users)
        #self._unlearn_min_centering_sparse(54,17,init_rmat,self.smat_unlearn_sparse)
        end = time.time()
        learn_unlearn_time = end - start
        print("Unlearn Supported Learning: {}".format(end-start))
        
        rows, cols, vals = self.smat_unlearn_sparse_csr
        self.smat_unlearn_sparse = sps.csr_matrix((vals,(rows,cols)),shape=(self.M,self.M))
        #print(self.smat_unlearn_sparse)

        start = time.time()
        #self._unlearn_min_centering_sparse(54,17,rmat_scipy,self.smat_unlearn_sparse)
        self._unlearn_min_centering_matrix(54,17,rmat_scipy,self.smat_unlearn_sparse,init_rmat)
        end = time.time()
        unlearn_time = end - start
        print("Unlearn: {}".format(end-start))
        
        start = time.time()
        rmat, item_means = self._mean_center(ratings, init_rmat, items, users)
           
        rmat = self._normalize(rmat)
        _logger.info('[%s] computing similarity matrix', self._timer)
        smat = self._compute_similarities(rmat,items,users)
        
        end = time.time()
        native_learn_time = end - start
        #print(smat.to_scipy())
        print("Native Learning: {}".format(end-start))

        _logger.info('[%s] got neighborhoods for %d of %d items',
                     self._timer, np.sum(np.diff(smat.rowptrs) > 0), n_items)

        _logger.info('[%s] computed %d neighbor pairs', self._timer, smat.nnz)
        
        self.item_index_ = items
        self.item_means_ = item_means
        self.item_counts_ = np.diff(smat.rowptrs)
        self.sim_matrix_ = smat
        self.user_index_ = users
        self.rating_matrix_ = init_rmat
        f = open("output_matrix.csv","a+")
        f.write("{},{},{},{}\n".format(init_rmat.nnz ,native_learn_time,learn_unlearn_time,unlearn_time))
        f.close()
        return self 

    def _compute_similarities_unlearn_min_centering(self,ratings,rmat,items,users):
        rmat_scipy = rmat.to_scipy()
        N = len(users)
        M = len(items)

        SUM_ITEM = np.zeros(M)
        Count_ITEM = np.zeros(M)
        MEAN_ITEM = np.zeros(M)
        
        S_ITEM = np.zeros((M,M))
        S_ITEMITEM = np.zeros((M,M))
        smat = np.zeros((M,M))
        Count_ITEMITEM = np.zeros((M,M))

        #rmat_scipy[18,22] = 0
        for i in range(N):
            for j in range(M):
                if rmat_scipy[i,j] != 0:
                    #if j == 22 or j == 76:
                        #print(i,j,rmat_scipy[i,j],items[i],users[j])
                    SUM_ITEM[j] += rmat_scipy[i,j]
                    Count_ITEM[j] += 1
        MEAN_ITEM = SUM_ITEM / Count_ITEM
        
        for k in range(M):
            for l in range(M):
                for i in range(N):
                    if rmat_scipy[i,l] != 0 and rmat_scipy[i,k] != 0:
                        #print(k,l)
                        S_ITEMITEM[k,l] += rmat_scipy[i,l] * rmat_scipy[i,k]
                        S_ITEM[k,l] += rmat_scipy[i,k]
                        Count_ITEMITEM[k,l] += 1
        
        self.S_I = S_ITEM
        self.S_II = S_ITEMITEM
        self.M_I = MEAN_ITEM
        self.N_I = Count_ITEM
        self.N_II = Count_ITEMITEM
        self.Sum_I = SUM_ITEM
        self.N = N
        self.M = M
        
        for k in range(M):
            for l in range(M):
                if Count_ITEMITEM[k,l] > 0:
                    smat_val = self._learn_sim(S_ITEMITEM[k,l],S_ITEMITEM[k,k],S_ITEMITEM[l,l],S_ITEM[k,l],S_ITEM[l,k],MEAN_ITEM[k],MEAN_ITEM[l],Count_ITEMITEM[k,l],Count_ITEM[k],Count_ITEM[l],SUM_ITEM[k],SUM_ITEM[l])
                    if smat_val > 0 and k != l:
                        smat[k,l] = smat_val
                        #print(k,l,S_ITEMITEM[k,l],S_ITEMITEM[k,k],S_ITEMITEM[l,l],S_ITEM[k,l],S_ITEM[l,k],MEAN_ITEM[k],MEAN_ITEM[l],Count_ITEMITEM[k,l],Count_ITEM[k],Count_ITEM[l],SUM_ITEM[k],SUM_ITEM[l])
                        #print(smat[k,l],k,l)
                    
        self.smat_unlearn = smat

    def _learn_sim(self,Skl,Skk,Sll,Sk,Sl,Mk,Ml,Nkl,Nk,Nl,Sumk,Suml):
        top = Skl-Mk*Sl-Ml*Sk+Mk*Ml*Nkl
        deno = np.sqrt(Skk-2*Mk*Sumk+(Mk**2)*Nk) * np.sqrt(Sll-2*Ml*Suml+(Ml**2)*Nl)
        if deno == 0:
            return 0
        else:
            return top/deno 

    def _learn_sim_vectorize(self, S_II=None, S_I=None, M_I=None, N_I=None, N_II=None, SUM_I=None):
        S_II=self.S_II_sparse
        S_I=self.S_I_sparse
        M_I=self.M_I_sparse
        N_I=self.N_I_sparse
        N_II=self.N_II_sparse
        SUM_I=self.Sum_I_sparse

        M_T = M_I.transpose()
        top = S_II - S_I.multiply(M_I) - S_I.transpose().multiply(M_T) + M_I.multiply(M_T).multiply(N_II)
        
        #print(S_II[22,76],S_I.multiply(M_T)[22,76],S_I.transpose().multiply(M_I)[22,76],M_I.multiply(M_T).multiply(N_II)[22,76])
        
        deno = sps.csr_matrix(S_II.diagonal()) - 2 * M_I.multiply(SUM_I) + M_I.multiply(M_I).multiply(N_I)
        #deno = 2 * M_I.multiply(SUM_I) + M_I.multiply(M_I).multiply(N_I)
        deno = deno.sqrt()
        deno = deno.multiply(deno.transpose())
        is_nz = deno > 0
        deno[is_nz] = np.reciprocal(deno[is_nz])
        smat = top.multiply(deno)

        
        smat = smat.tocoo()

        rows, cols, vals = smat.row, smat.col, smat.data

        #rows = rows[:smat.nnz]
        #cols = cols[:smat.nnz]
        #vals = vals[:smat.nnz]

        
        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        
        
        return rows, cols, vals #sps.csr_matrix((vals,(rows,cols)),shape=(self.M,self.M))

    def _learn_sim_global_vectorize(self, S_II=None, S_I=None, M_I=None, N_I=None, N_II=None, SUM_I=None, g=None, UM=None):
        S_II=self.S_II_sparse
        S_I=self.S_I_sparse
        M_I=self.M_I_sparse
        N_I=self.N_I_sparse
        N_II=self.N_II_sparse
        SUM_I=self.Sum_I_sparse
        UM = self.UM_I
        g = sps.csr_matrix([self.G])
        
        M_T = M_I.transpose()
        
        M_I_G = sps.csr_matrix(self.M_I - self.G)
        M_I_M_I = sps.csr_matrix( np.repeat(self.M_I,self.M,axis = 0) + np.repeat(self.M_I.T,self.M,axis=1))
        top = S_II - S_I.multiply(M_I) - S_I.transpose().multiply(M_T) + g.multiply(S_I+S_I.transpose()) - g.multiply(M_I_M_I).multiply(N_II) + M_I.multiply(M_T).multiply(N_II) + g.multiply(g).multiply(N_II)
        deno = sps.csr_matrix(S_II.diagonal()) - 2 * (M_I_G).multiply(SUM_I) + (M_I_G).multiply(M_I_G).multiply(N_I)
        #deno = 2 * M_I.multiply(SUM_I) + M_I.multiply(M_I).multiply(N_I)
        deno = deno.sqrt()
        print(deno.shape,deno[0,1],S_II[1,1],M_I_G[0,1],SUM_I[0,1],N_I[0,1],S_I[1,1])
        deno = deno.multiply(deno.transpose())
        is_nz = deno > 0
        deno[is_nz] = np.reciprocal(deno[is_nz])
        smat = top.multiply(deno)
        
        return smat

    def _unlearn_min_centering(self,u,t,rmat,smat):
        rmat_scipy = rmat.to_scipy()
        
        self.Sum_I[t] -= rmat_scipy[u,t]
        self.M_I[t] = (self.M_I[t] * self.N_I[t] - rmat_scipy[u,t]) / (self.N_I[t] - 1)
        self.N_I[t] -= 1

        
        for k in range(self.M):
            for l in range(self.M):
                if rmat_scipy[u,k] != 0 and rmat_scipy[u,l] != 0:
                    if k == t or l == t:
                        #print(k,l)
                        self.S_II[k,l] -= rmat_scipy[u,k] * rmat_scipy[u,l]
                        self.S_I[k,l] -= rmat_scipy[u,k]
                        self.N_II[k,l] -= 1
                
        for k in range(self.M):
            if smat[k,t] != 0:
                #print(smat[k,t],k,t)
                smat[k,t] = self._learn_sim(self.S_II[k,t],self.S_II[k,k],self.S_II[t,t],self.S_I[k,t],self.S_I[t,k],self.M_I[k],self.M_I[t],self.N_II[k,t],self.N_I[k],self.N_I[t],self.Sum_I[k],self.Sum_I[t])
                smat[t,k] = smat[k,t]
                #print(smat[k,t])

    def _compute_similarities_unlearn(self,ratings,rmat,items,users):
        
        rmat_scipy = rmat.to_scipy()
        
        N = len(users)
        M = len(items)

        SUM_USER = np.zeros(N)
        MEAN_USER = np.zeros(N)
        SUM_ITEM = np.zeros(M)
        Count_ITEM = np.zeros(M)
        MEAN_ITEM = np.zeros(M)
        SUM_g = 0
        
        for i in range(N):
            Count = 0
            for j in range(M):
                if rmat_scipy[i,j] != 0:
                    SUM_USER[i] += rmat_scipy[i,j]
                    Count += 1
                    SUM_ITEM[j] += rmat_scipy[i,j]
                    Count_ITEM[j] += 1
                    SUM_g += rmat_scipy[i,j]
            MEAN_USER[i] = SUM_USER[i] / Count
        
        g = SUM_g / np.sum(Count_ITEM)
        #print(np.sum(Count_ITEM))
        
        S_Item = np.zeros(len(items))
        S_ItemItem = np.zeros((len(items),len(items)))
        smat = np.zeros((len(items),len(items)))
        
        r_copy = rmat_scipy.copy()
        for k in range(M):
            MEAN_ITEM[k] = SUM_ITEM[k] / Count_ITEM[k]
            for i in range(N):
                if rmat_scipy[i,k] != 0:
                    S_Item[k] += rmat_scipy[i,k] - MEAN_USER[i]
            for l in range(M):
                for i in range(N):
                    if rmat_scipy[i,k] != 0 and rmat_scipy[i,l] != 0:
                        S_ItemItem[k,l] += (rmat_scipy[i,k]-MEAN_USER[i])*(rmat_scipy[i,l]-MEAN_USER[i])
                #smat[k,l] = self._learn_similarities_(S_ItemItem[k,l],S_Item[k],S_Item[l],S_ItemItem[k,k],S_ItemItem[l,l],g,MEAN_ITEM[k],MEAN_ITEM[l],N)
                #if S_ItemItem[k,l] != 0 and k!=l:
                #    print(k,l,smat[k,l] )
        for k in range(M):
            tmp = np.sqrt(S_ItemItem[k,k] - 2 * (MEAN_ITEM[k] - g) * S_Item[k] + ((MEAN_ITEM[k] - g)**2) * Count_ITEM[k])
            if tmp != 0:
                print(tmp,k,S_ItemItem[k,k],MEAN_ITEM[k],Count_ITEM[k],g)
            for l in range(M):
        #        if k != l:
                smat[k,l] = self._learn_similarities_(S_ItemItem[k,l],S_Item[k],S_Item[l],S_ItemItem[k,k],S_ItemItem[l,l],g,MEAN_ITEM[k],MEAN_ITEM[l],Count_ITEM[k],Count_ITEM[l])
        #            if S_ItemItem[k,l] != 0:
        #                print(k,l,smat[k,l], (r_copy[:,k].T @ r_copy[:, l]) )
                #if rmat_scipy[k,l] != 0:
                #    print(rmat_scipy)
        #print(smat)
    
    def _learn_similarities_(self,Skl,Sk,Sl,Skk,Sll,g,Mk,Ml,N1,N2):
        
        top = Skl - Mk*Sl - Ml*Sk + g*(Sk+Sl) - g * (Mk + Ml) * N1 + Mk * Ml * N1 + g*g*N1
        down = np.sqrt(Skk - 2 * (Mk - g) * Sk + ((Mk - g)**2) * N1) 
        down*= np.sqrt(Sll - 2 * (Ml - g) * Sl + ((Ml - g)**2) * N2)
        if down == 0:
            return 0
        return top / down
    
    def _compute_similarities_unlearn_min_centering_sparse_slow(self,rmat_scipy,items,users):
        
        rmat_coo = rmat_scipy.tocoo()
        rows, cols, vals = rmat_coo.row, rmat_coo.col, rmat_coo.data        
        N = len(users)
        M = len(items)

        SUM_ITEM = np.zeros(M)
        Count_ITEM = np.zeros(M)
        MEAN_ITEM = np.zeros(M)
        
        Count_ITEMITEM_data = []
        S_ITEM_data = []
        S_ITEMITEM_data = []

        II_ROWS, II_COLS = [], []
        for i in range(rmat_scipy.nnz):
            c, v = cols[i], vals[i]
            SUM_ITEM[c] += v
            Count_ITEM[c] += 1

        MEAN_ITEM = SUM_ITEM / Count_ITEM
        for i in range(N):
            idx = np.argwhere(rows == i)
            for k_idx in range(len(idx)):
                for l_idx in range(len(idx)):
                    k = cols[idx[k_idx]][0]
                    l = cols[idx[l_idx]][0]
                    II_ROWS.append(k)
                    II_COLS.append(l)
                    Count_ITEMITEM_data.append(1)
                    s_ii = vals[idx[k_idx]][0] * vals[idx[l_idx]][0]
                    S_ITEMITEM_data.append(s_ii)
                    S_ITEM_data.append(vals[idx[k_idx]][0])
        II_ROWS = np.array(II_ROWS)#.flatten()
        II_COLS = np.array(II_COLS)#.flatten()
        S_ITEM_data = np.array(S_ITEM_data)#.flatten()
        S_ITEMITEM_data = np.array(S_ITEMITEM_data)#.flatten()
        
        Count_ITEMITEM = sps.csr_matrix((Count_ITEMITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        S_ITEM = sps.csr_matrix((S_ITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        S_ITEMITEM = sps.csr_matrix((S_ITEMITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        
        self.S_I_sparse = S_ITEM
        self.S_II_sparse = S_ITEMITEM
        self.N_II_sparse = Count_ITEMITEM

        self.M_I = MEAN_ITEM
        self.N_I = Count_ITEM
        self.Sum_I = SUM_ITEM

        self.M_I_sparse = sps.csr_matrix(MEAN_ITEM)
        self.N_I_sparse = sps.csr_matrix(Count_ITEM)
        self.Sum_I_sparse = sps.csr_matrix(SUM_ITEM)

        self.N = N
        self.M = M
        
        self.smat_unlearn_sparse_csr = self._learn_sim_vectorize()
        
    
    def _compute_similarities_unlearn_min_centering_sparse_vectorize(self,rmat_scipy,items,users):
        #rmat_coo = rmat_scipy.tocoo()
        #cols, vals = rmat_coo.col, rmat_coo.data        
        N = len(users)
        M = len(items)

        #SUM_ITEM = np.zeros(M)
        #Count_ITEM = np.zeros(M)
        #MEAN_ITEM = np.zeros(M)
        #Count_ITEMITEM_data = []
        #S_ITEM_data = []
        #S_ITEMITEM_data = []

        #II_ROWS, II_COLS = [], []
        '''
        for i in range(rmat_scipy.nnz):
            c, v = cols[i], vals[i]
            SUM_ITEM[c] += v
            Count_ITEM[c] += 1

        MEAN_ITEM = SUM_ITEM / Count_ITEM
        '''
        
        rmat_mask = rmat_scipy.copy()
        rmat_mask[rmat_scipy>0] = 1

        self.S_I_sparse = rmat_scipy.transpose() @ rmat_mask
        self.S_II_sparse = rmat_scipy.transpose() @ rmat_scipy
        self.N_II_sparse = rmat_mask.transpose() @ rmat_mask

        #
        
        #self.S_I_sparse.sort_indices()
        print("self.S_I_sparse.indices[self.S_I_sparse.indptr[138]:self.S_I_sparse.indptr[139]]: ",self.S_I_sparse.indices[self.S_I_sparse.indptr[138]:self.S_I_sparse.indptr[139]])
        print("self.S_I_sparse.data[self.S_I_sparse.indptr[138]:self.S_I_sparse.indptr[139]]: ",self.S_I_sparse.data[self.S_I_sparse.indptr[138]:self.S_I_sparse.indptr[139]])
        print("self.S_I_sparse[138,17]",self.S_I_sparse[138,17])
        #print("self.S_I_sparse.getrow(138)")
        #print(self.S_I_sparse.getrow(138))
        print("self.S_I_sparse.indices[self.S_I_sparse.indptr[17]:self.S_I_sparse.indptr[18]],self.S_I_sparse[17,138]: ",self.S_I_sparse.indices[self.S_I_sparse.indptr[17]:self.S_I_sparse.indptr[18]])
        print("self.S_I_sparse.data[self.S_I_sparse.indptr[17]:self.S_I_sparse.indptr[18]]: ",self.S_I_sparse.data[self.S_I_sparse.indptr[17]:self.S_I_sparse.indptr[18]])
        print("self.S_I_sparse[17,138]",self.S_I_sparse[17,138])
        #print("self.S_I_sparse.getrow(17): ")
        #print(self.S_I_sparse.getrow(17))
        
        self.N_I = np.array(rmat_mask.sum(axis = 0)).squeeze()
        self.Sum_I = np.array(rmat_scipy.sum(axis=0)).squeeze()
        self.M_I = self.Sum_I / self.N_I

        self.M_I_sparse = sps.csr_matrix(self.M_I)
        self.N_I_sparse = sps.csr_matrix(self.N_I)
        self.Sum_I_sparse = sps.csr_matrix(self.Sum_I)

        self.N = N
        self.M = M
        
        self.smat_unlearn_sparse_csr = self._learn_sim_vectorize()

    def _compute_similarities_unlearn_min_centering_sparse(self,rmat_scipy,items,users):
        
        rmat_coo = rmat_scipy.tocoo()
        rows, cols, vals = rmat_coo.row, rmat_coo.col, rmat_coo.data        
        N = len(users)
        M = len(items)
        SUM_ITEM = np.zeros(M)
        Count_ITEM = np.zeros(M)
        MEAN_ITEM = np.zeros(M)
        
        Count_ITEMITEM_data = []
        S_ITEM_data = []
        S_ITEMITEM_data = []

        II_ROWS, II_COLS = [], []
        for i in range(rmat_scipy.nnz):
            c, v = cols[i], vals[i]
            SUM_ITEM[c] += v
            Count_ITEM[c] += 1

        MEAN_ITEM = SUM_ITEM / Count_ITEM
        for i in range(N):
            idx = np.argwhere(rows == i)
            for k_idx in range(len(idx)):
                for l_idx in range(len(idx)):
                    k = cols[idx[k_idx]][0]
                    l = cols[idx[l_idx]][0]
                    II_ROWS.append(k)
                    II_COLS.append(l)
                    Count_ITEMITEM_data.append(1)
                    S_ITEM_data.append(rmat_scipy[i,k])
                    s_ii = rmat_scipy[i,k] * rmat_scipy[i,l]
                    S_ITEMITEM_data.append(s_ii)
        II_ROWS = np.array(II_ROWS)
        II_COLS = np.array(II_COLS)
        S_ITEM_data = np.array(S_ITEM_data)
        S_ITEMITEM_data = np.array(S_ITEMITEM_data)
        
        Count_ITEMITEM = sps.csr_matrix((Count_ITEMITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        S_ITEM = sps.csr_matrix((S_ITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        S_ITEMITEM = sps.csr_matrix((S_ITEMITEM_data, (II_ROWS,II_COLS)), shape=(M,M))
        
        self.S_I_sparse = S_ITEM
        self.S_II_sparse = S_ITEMITEM
        self.N_II_sparse = Count_ITEMITEM

        self.M_I = MEAN_ITEM
        self.N_I = Count_ITEM
        self.Sum_I = SUM_ITEM

        self.M_I_sparse = sps.csr_matrix(MEAN_ITEM)
        self.N_I_sparse = sps.csr_matrix(Count_ITEM)
        self.Sum_I_sparse = sps.csr_matrix(SUM_ITEM)

        self.N = N
        self.M = M
        
        self.smat_unlearn_sparse_csr = self._learn_sim_vectorize()
    
    def _unlearn_min_centering_sparse(self,u,t,rmat_scipy,smat):
        #print(type(self.Sum_I),type(self.M_I),type(self.N_I),type(self.S_II_sparse),type(self.S_I_sparse),type(self.S_II_sparse))
        val_u_t = rmat_scipy[u,t]
        '''
        self.Sum_I[0,t] -= val_u_t
        self.M_I[0,t] = ( self.M_I[0,t] * self.N_I[0,t] - val_u_t ) / (self.N_I[0,t] - 1)
        self.N_I[0,t] -= 1
        '''
        self.Sum_I[t] -= val_u_t
        self.M_I[t] = ( self.M_I[t] * self.N_I[t] - val_u_t ) / (self.N_I[t] - 1)
        self.N_I[t] -= 1
        for l in self.N_II_sparse.getrow(t).indices:
            #k, l = rows[i], cols[i]
            val_u_l = rmat_scipy[u,l]
            if val_u_l > 0:
                self.S_II_sparse[t,l] -= val_u_t * val_u_l
                self.S_I_sparse[t,l] -= val_u_t
                self.N_II_sparse[t,l] -= 1
                #print(val_u_t * val_u_l,val_u_t,t,l)
                if t != l:
                    self.S_II_sparse[l,t] = self.S_II_sparse[t,l]
                    self.S_I_sparse[l,t] = self.S_I_sparse[t,l]
                    self.N_II_sparse[l,t] -= 1
        '''
        val_u_ls = rmat_scipy[u,:]
        val_u_ls_mask = val_u_ls.copy()
        val_u_ls_mask[val_u_ls > 0] = 1
        self.S_II_sparse[t,:] -= val_u_t*val_u_ls.multiply(val_u_ls_mask)
        self.S_I_sparse[t,:] -= val_u_t*val_u_ls_mask
        self.N_II_sparse[t,:] -= val_u_ls_mask
        '''
        #indptr = self.N_II_sparse.indptr
        #rmat_indptr = rmat_scipy.indptr
        #rmat_indices = rmat_scipy.indices[rmat_indptr[u]:rmat_indptr[u+1]]

        for k in smat.getrow(t).indices:
            if k != t:
                #smat[k,t] = self._learn_sim(self.S_II_sparse[k,t],self.S_II_sparse[k,k],self.S_II_sparse[t,t],self.S_I_sparse[k,t],self.S_I_sparse[t,k],self.M_I[0,k],self.M_I[0,t],self.N_II_sparse[k,t],self.N_I[0,k],self.N_I[0,t],self.Sum_I[0,k],self.Sum_I[0,t])
                smat[k,t] = self._learn_sim(self.S_II_sparse[k,t],self.S_II_sparse[k,k],self.S_II_sparse[t,t],self.S_I_sparse[k,t],self.S_I_sparse[t,k],self.M_I[k],self.M_I[t],self.N_II_sparse[k,t],self.N_I[k],self.N_I[t],self.Sum_I[k],self.Sum_I[t])
                #print(self.S_II_sparse[k,t],self.S_II_sparse[k,k],self.S_II_sparse[t,t],self.S_I_sparse[k,t],self.S_I_sparse[t,k],self.M_I[0,k],self.M_I[0,t],self.N_II_sparse[k,t],self.N_I[0,k],self.N_I[0,t],self.Sum_I[0,k],self.Sum_I[0,t])
                smat[t,k] = smat[k,t]
            #print(smat[k,t],k,t)
        #self.S_I_sparse.eliminate_zeros()
        #self.S_II_sparse.eliminate_zeros()
        #self.N_II_sparse.eliminate_zeros()
        #smat.eliminate_zeros()

    def _compute_similarities_unlearn_min_centering_matrix_vectorize(self,rmat_scipy,items,users):
        N = len(users)
        M = len(items)
        rmat_mask = rmat_scipy.copy()
        rmat_mask[rmat_scipy>0] = 1

        self.S_I_sparse = rmat_scipy.transpose() @ rmat_mask
        self.S_II_sparse = rmat_scipy.transpose() @ rmat_scipy
        self.N_II_sparse = rmat_mask.transpose() @ rmat_mask

        self.S_I_matrix = matrix.CSR(self.S_I_sparse.shape[0], self.S_I_sparse.shape[1], self.S_I_sparse.nnz,
                                        self.S_I_sparse.indptr.copy(), self.S_I_sparse.indices.copy(), self.S_I_sparse.data)
        self.S_II_matrix = matrix.CSR(self.S_II_sparse.shape[0], self.S_II_sparse.shape[1], self.S_II_sparse.nnz,
                                        self.S_II_sparse.indptr.copy(), self.S_II_sparse.indices.copy(), self.S_II_sparse.data)
        self.N_II_matrix = matrix.CSR(self.N_II_sparse.shape[0], self.N_II_sparse.shape[1], self.N_II_sparse.nnz,
                                        self.N_II_sparse.indptr.copy(), self.N_II_sparse.indices.copy(), self.N_II_sparse.data)

        self.N_I = rmat_mask.sum(axis = 0)
        self.Sum_I = rmat_scipy.sum(axis=0)
        self.M_I = self.Sum_I / self.N_I

        self.M_I_sparse = sps.csr_matrix(self.M_I)
        self.N_I_sparse = sps.csr_matrix(self.N_I)
        self.Sum_I_sparse = sps.csr_matrix(self.Sum_I)

        self.N = N
        self.M = M
        
        self.smat_unlearn_sparse_csr = self._learn_sim_vectorize()

    def _unlearn_min_centering_matrix(self,u,t,rmat_scipy,smat,rmat):
        
        val_u_t = rmat_scipy[u,t]

        self.Sum_I[0,t] -= val_u_t
        self.M_I[0,t] = ( self.M_I[0,t] * self.N_I[0,t] - val_u_t ) / (self.N_I[0,t] - 1)
        self.N_I[0,t] -= 1

        rmat_u_colinds = rmat.row_cs(u)
        for l_idx, l in enumerate(self.S_I_matrix.row_cs(t)):
            rmat_idx = np.where(rmat_u_colinds == l)[0]
            if len(rmat_idx == 1):
                val_u_l = rmat.values[rmat.rowptrs[u]+rmat_idx]
                self.S_II_matrix.values[self.S_I_matrix.rowptrs[t]+l_idx] -= val_u_l * val_u_t
                self.S_I_matrix.values[self.S_II_matrix.rowptrs[t]+l_idx] -= val_u_t
                self.N_II_matrix.values[self.N_II_matrix.rowptrs[t]+l_idx] -= 1
        
        
        
        for k_idx, k in enumerate(smat.getrow(t).indices):
            if k != t:
                #print(k,t,smat[k,t])
                indices_k = self.N_II_matrix.colinds[self.N_II_matrix.rowptrs[k]:self.N_II_matrix.rowptrs[k+1]]
                
                kt_idx = self.N_II_matrix.rowptrs[k] + np.where(indices_k == t)[0]
                kk_idx = self.N_II_matrix.rowptrs[k] + np.where(indices_k == k)[0]

                indices_t = self.N_II_matrix.colinds[self.N_II_matrix.rowptrs[t]:self.N_II_matrix.rowptrs[t+1]]
                tk_idx = self.N_II_matrix.rowptrs[t] + np.where(indices_t == k)[0]
                tt_idx = self.N_II_matrix.rowptrs[t] + np.where(indices_t == t)[0]

                smat.data[smat.indptr[t]+k_idx] = self._learn_sim(
                                                                    self.S_II_matrix.values[kt_idx],
                                                                    self.S_II_matrix.values[kk_idx],
                                                                    self.S_II_matrix.values[tt_idx],
                                                                    self.S_I_matrix.values[kt_idx],
                                                                    self.S_I_matrix.values[tk_idx],
                                                                    self.M_I[0,k],
                                                                    self.M_I[0,t],
                                                                    self.N_II_matrix.values[kt_idx],
                                                                    self.N_I[0,k],
                                                                    self.N_I[0,t],
                                                                    self.Sum_I[0,k],
                                                                    self.Sum_I[0,t])
                
                
                #print(self.S_I_matrix.values[kt_idx],self.S_I_sparse[k,t],self.S_I_matrix.row_cs(k),self.S_I_matrix.row_vs(k))
                #print(self.S_I_matrix.values[tk_idx],self.S_I_sparse[t,k],self.S_I_matrix.row_cs(t),self.S_I_matrix.row_vs(t))
                
                smat[k,t] = smat[t,k]
                #print(smat.data[smat.indptr[t]+k_idx])
        #self.S_I_sparse.eliminate_zeros()
        #self.S_II_sparse.eliminate_zeros()
        #self.N_II_sparse.eliminate_zeros()
        smat.eliminate_zeros()

    def _compute_similarities_unlearn_global_centering_matrix_vectorize(self,rmat_scipy,items,users):
        M = len(items)
        N = len(users)
        rmat_mask = rmat_scipy.copy()
        rmat_mask[rmat_scipy>0] = 1

        self.N_I = rmat_mask.sum(axis = 0)
        Sum_I = rmat_scipy.sum(axis=0)
        
        self.M_I = Sum_I / self.N_I
        self.UM_I = rmat_scipy.mean(axis=1)
        self.G = rmat_scipy.sum() / rmat_scipy.nnz

        self.Sum_I = np.multiply((rmat_scipy - self.M_I - self.UM_I + self.G),rmat_mask.toarray()).sum(axis=0)
        
        self.M_I_sparse = sps.csr_matrix(self.M_I)
        self.N_I_sparse = sps.csr_matrix(self.N_I)
        self.Sum_I_sparse = sps.csr_matrix(self.Sum_I)
        self.UM_I_sparse = sps.csr_matrix(self.UM_I)

        self.N = N
        self.M = M

        #print(rmat_scipy.shape,self.UM_I_sparse.shape,self.UM_I.shape)
        rmat_centered = sps.csr_matrix(rmat_scipy.toarray() - self.UM_I)
        rmat_centered = rmat_centered.multiply(rmat_mask)
        self.S_I_sparse = rmat_centered.transpose() @ rmat_mask
        self.S_II_sparse = rmat_centered.transpose() @ rmat_centered
        self.N_II_sparse = rmat_mask.transpose() @ rmat_mask

        self.smat_unlearn_sparse_csr = self._learn_sim_global_vectorize()


    def _mean_center(self, ratings, rmat, items, users):
        if not self.center:
            return rmat, None

        item_means = ratings.groupby('item').rating.mean()
        item_means = item_means.reindex(items).values
        
        user_means = ratings.groupby('user').rating.mean()
        user_means = user_means.reindex(users).values

        global_mean = ratings.rating.mean()

        #mcvals = rmat.values - item_means[rmat.colinds] - user_means[rmat.rowinds()] + global_mean
        #Old Mean Centering
        mcvals = rmat.values - item_means[rmat.colinds]
        
        nmat = matrix.CSR(rmat.nrows, rmat.ncols, rmat.nnz,
                          rmat.rowptrs.copy(), rmat.colinds.copy(), mcvals)
        _logger.info('[%s] computed means for %d items', self._timer, len(item_means))
        return nmat, item_means

    def _normalize(self, rmat):
        rmat = rmat.to_scipy()
        # compute column norms
        norms = spla.norm(rmat, 2, axis=0)
        
        # and multiply by a diagonal to normalize columns
        recip_norms = norms.copy()
        is_nz = recip_norms > 0
        #print(recip_norms[1],rmat.getcol(1))
        recip_norms[is_nz] = np.reciprocal(recip_norms[is_nz])
        norm_mat = rmat @ sps.diags(recip_norms)
        assert norm_mat.shape[1] == rmat.shape[1]
        # and reset NaN
        norm_mat.data[np.isnan(norm_mat.data)] = 0
        _logger.info('[%s] normalized rating matrix columns', self._timer)
        return matrix.CSR.from_scipy(norm_mat, False)

    def _compute_similarities(self, rmat, items, users):
        mkl = matrix.mkl_ops()
        mkl = None
        if mkl is None:
            return self._scipy_similarities(rmat,items,users)
        else:
            return self._mkl_similarities(mkl, rmat)

    def _scipy_similarities(self, rmat,items,users):
        
        nitems = rmat.ncols
        sp_rmat = rmat.to_scipy()
        #print(sp_rmat.tocoo())
        _logger.info('[%s] multiplying matrix with scipy', self._timer)
        smat = sp_rmat.T @ sp_rmat
        smat = smat.tocoo()
        #print(smat)
        rows, cols, vals = smat.row, smat.col, smat.data

        rows = rows[:smat.nnz]
        cols = cols[:smat.nnz]
        vals = vals[:smat.nnz]

        
        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        csr = self._select_similarities(nitems, rows, cols, vals)
        return csr

    def _mkl_similarities(self, mkl, rmat):
        nitems = rmat.ncols
        assert rmat.values is not None

        _logger.info('[%s] multiplying matrix with MKL', self._timer)
        smat = mkl.csr_syrk(rmat)
        rows = smat.rowinds()
        cols = smat.colinds
        vals = smat.values

        rows, cols, vals = self._filter_similarities(rows, cols, vals)
        del smat
        nnz = len(rows)

        _logger.info('[%s] making matrix symmetric (%d nnz)', self._timer, nnz)
        rows = np.resize(rows, nnz * 2)
        cols = np.resize(cols, nnz * 2)
        vals = np.resize(vals, nnz * 2)
        rows[nnz:] = cols[:nnz]
        cols[nnz:] = rows[:nnz]
        vals[nnz:] = vals[:nnz]

        csr = self._select_similarities(nitems, rows, cols, vals)
        return csr

    def _filter_similarities(self, rows, cols, vals):
        "Threshold similarites & remove self-similarities."
        _logger.info('[%s] filtering %d similarities', self._timer, len(rows))
        # remove self-similarity
        mask = rows != cols

        # remove too-small similarities
        if self.min_sim is not None:
            mask = np.logical_and(mask, vals >= self.min_sim)

        _logger.info('[%s] filter keeps %d of %d entries', self._timer, np.sum(mask), len(rows))

        return rows[mask], cols[mask], vals[mask]

    def _select_similarities(self, nitems, rows, cols, vals):
        _logger.info('[%s] ordering similarities', self._timer)
        csr = matrix.CSR.from_coo(rows, cols, vals, shape=(nitems, nitems))
        csr.sort_values()

        if self.save_nbrs is None or self.save_nbrs <= 0:
            return csr

        _logger.info('[%s] picking %d top similarities', self._timer, self.save_nbrs)
        counts = csr.row_nnzs()
        _logger.debug('have %d rows in size range [%d,%d]',
                      len(counts), np.min(counts), np.max(counts))
        ncounts = np.fmin(counts, self.save_nbrs)
        _logger.debug('will have %d rows in size range [%d,%d]',
                      len(ncounts), np.min(ncounts), np.max(ncounts))
        assert np.all(ncounts <= self.save_nbrs)
        assert np.all(ncounts >= 0)
        nnz = np.sum(ncounts)

        rp2 = np.zeros_like(csr.rowptrs)
        rp2[1:] = np.cumsum(ncounts)
        ci2 = np.zeros(nnz, np.int32)
        vs2 = np.zeros(nnz)
        for i in range(nitems):
            sp1 = csr.rowptrs[i]
            sp2 = rp2[i]

            ep1 = sp1 + ncounts[i]
            ep2 = sp2 + ncounts[i]
            assert ep1 - sp1 == ep2 - sp2

            ci2[sp2:ep2] = csr.colinds[sp1:ep1]
            vs2[sp2:ep2] = csr.values[sp1:ep1]

        return matrix.CSR(csr.nrows, csr.ncols, nnz, rp2, ci2, vs2)

    def predict_for_user(self, user, items, ratings=None):
        _logger.debug('predicting %d items for user %s', len(items), user)
        if ratings is None:
            if user not in self.user_index_:
                _logger.debug('user %s missing, returning empty predictions', user)
                return pd.Series(np.nan, index=items)
            upos = self.user_index_.get_loc(user)
            ratings = pd.Series(self.rating_matrix_.row_vs(upos),
                                index=pd.Index(self.item_index_[self.rating_matrix_.row_cs(upos)]))

        if not ratings.index.is_unique:
            wmsg = 'user {} has duplicate ratings, this is likely to cause problems'.format(user)
            warnings.warn(wmsg, DataWarning)

        # set up rating array
        # get rated item positions & limit to in-model items
        ri_pos = self.item_index_.get_indexer(ratings.index)
        m_rates = ratings[ri_pos >= 0]
        ri_pos = ri_pos[ri_pos >= 0]
        rate_v = np.full(len(self.item_index_), np.nan, dtype=np.float_)
        # mean-center the rating array
        if self.center:
            rate_v[ri_pos] = m_rates.values - self.item_means_[ri_pos]
        else:
            rate_v[ri_pos] = m_rates.values
        _logger.debug('user %s: %d of %d rated items in model', user, len(ri_pos), len(ratings))
        assert np.sum(np.logical_not(np.isnan(rate_v))) == len(ri_pos)

        # set up item result vector
        # ipos will be an array of item indices
        i_pos = self.item_index_.get_indexer(items)
        i_pos = i_pos[i_pos >= 0]
        _logger.debug('user %s: %d of %d requested items in model', user, len(i_pos), len(items))

        # scratch result array
        iscore = np.full(len(self.item_index_), np.nan, dtype=np.float_)

        # now compute the predictions
        iscore = self._predict_agg(self.sim_matrix_.N,
                                   len(self.item_index_),
                                   (self.min_nbrs, self.nnbrs),
                                   rate_v, i_pos)

        nscored = np.sum(np.logical_not(np.isnan(iscore)))
        if self.center:
            iscore += self.item_means_
        assert np.sum(np.logical_not(np.isnan(iscore))) == nscored

        results = pd.Series(iscore, index=self.item_index_)
        results = results[results.notna()]
        results = results.reindex(items, fill_value=np.nan)
        assert results.notna().sum() == nscored

        _logger.debug('user %s: predicted for %d of %d items',
                      user, results.notna().sum(), len(items))

        return results

    def __str__(self):
        return 'ItemItem(nnbrs={}, msize={})'.format(self.nnbrs, self.save_nbrs)
