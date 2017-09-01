import pandas as pd
import numpy as np
import graphlab
from collections import defaultdict
import gc
from sparse_soft_impute import SoftImpute
from time import time
from sklearn.preprocessing.imputation import Imputer
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
#from fancyimpute import BiScaler
from sparse_biscale import BiScale
import random

#from keras.regularizers import l2
#from keras.optimizers import SGD, RMSprop, Adam
#from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
#from keras.models import sequential, model

class MaskData(object):
    def __init__(self, mat, k):
        self.mat = mat
        self.k = k
        self.original = mat.copy()
        self.state = 31 # this is the random state for train test split
        self.test_data = None
        self.col_idx = None 
        self.row_idx = None 


    def _mask_clip(self, row_or_col):
        ''' 
        Cuts out items from matrix that do not contain at least k values on axis=0
        '''
        mat = self.mat
        k = self.k
        lil = mat.tolil()
        to_remove = []
        for idx, i in enumerate(lil.rows):
            if len(i) < k:
                to_remove.append(idx)
        lil.rows = np.delete(lil.rows, to_remove)
        lil.data = np.delete(lil.data, to_remove)
        if row_or_col == 'row':
            self.row_idx = np.delete(range(lil.shape[0]), to_remove) 
        elif row_or_col == 'col':
            self.col_idx = np.delete(range(lil.shape[0]), to_remove)
        remaining = lil.shape[0] - len(to_remove)
        lil = lil[:remaining]
        self.mat = lil
        return self 

    def _mask_find_subset(self):
        # Cut out users that don't meet minimum k
        self._mask_clip('row')

        self.mat = self.mat.T

        # Cut out items that don't meet minimum k
        self._mask_clip('col')

        self.mat = self.mat.T
        return self 

    def get_test_data(self):
        self._mask_find_subset()
        state = self.state
        k = self.k
        r_initial_idx = self.row_idx
        c_initial_idx = self.col_idx
        random.seed(state)
        mat = self.mat
        mat = mat.tocoo()
        mat_row, mat_col = mat.row, mat.col
        # we need to convert these numbers back to the original dataset's column and row indices
        mat_zip = zip(mat_row, mat_col)
        for idx, (row_val, col_val) in enumerate(mat_zip):
            r_val = r_initial_idx[row_val]
            c_val = c_initial_idx[col_val]
            mat_zip[idx] = (r_val, c_val) 

        self.test_data = random.sample(mat_zip, k) 
        return self

    def _convert_test_points(self, zipped_test):
        c_res = np.empty(len(zipped_test))
        r_res = np.empty(len(zipped_test)) 

        for idx, (r, c) in enumerate(zipped_test):
            c_res[idx] = c
            r_res[idx] = r

        return r_res.astype(int), c_res.astype(int)
           
    def mask(self):
        self.get_test_data()
        test_data = self.test_data
        original_data = self.original.tocoo()
        original_data = zip(original_data.row, original_data.col)
        mask = np.ones(self.original.nnz)
        for idx, item in enumerate(original_data):
            if item in test_data:
                mask[idx] = 0
        train_set = self.original.copy()
        train_set.data[mask == 0] = 0 
        train_set.eliminate_zeros()
        test_set = self.original
        print(test_data)
        test_data = self._convert_test_points(test_data)
        return train_set, self.original, test_data 


def load_movielens(path, leave_n_out=3):
    np.random.seed(0)
    df = pd.read_csv(path).sample(1000, random_state=2)   
    n = len(df)
    row = np.array(df.userId)
    col = np.array(df.movieId)
    data = np.array(df.rating)
    mat = coo_matrix((data, (row, col)))
    masker = MaskData(mat, leave_n_out)
    train, test, test_data_points = masker.mask()
    return train, test, test_data_points 

#def biscale(train):
#    bisc = BiScaler()
#    return bisc.fit_transform(train)
def gpredict(train, test_data_points):
    c = coo_matrix(train)
    sf = graphlab.SFrame({'row': c.row, 'col': c.col, 'data': c.data})
    sf_small = sf.dropna('data', how="all")
    sf_topredict = graphlab.SFrame({'row': test_data_points[0], 'col': test_data_points[1]})
    m1 = graphlab.factorization_recommender.create(sf_small, user_id='row', item_id='col', target='data', verbose=False)
    spred = m1.predict(sf_topredict)
    pred = spred.to_numpy()
    return pred

def my_predict(train, test_data_points):
    #train = csr_matrix(train)
    bs = BiScale(train)
    train = bs.solve()
    si = SoftImpute(max_rank=4, shrinkage_value=1, fill_method='sparse')
    si.complete(train)
    print(si._pred_sparse(test_data_points[0], test_data_points[1], si.X_filled))
    pred = si.predict_many(test_data_points[0], test_data_points[1]) 
    return pred

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

def neural_net(train):
    small_train = train.copy()
    small_train[[np.isnan(small_train)]] = 0
    c = coo_matrix(small_train)
    n_factors = 50
    n_users = train.shape[0]
    n_movies  = train.shape[1]
    user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
    movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)
    x = merge([u, m], mode='concat')
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(70, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(1)(x)
    nn = Model([user_in, movie_in], x)
    nn.compile(Adam(0.001), loss='mse')
    nn.fit([c.row, c.col], c.data, batch_size=64, nb_epoch=8)
    d = coo_matrix(train)
    pred = nn.predict_on_batch([d.row, d.col])
    pred = pred.reshape(train.shape[0], train.shape[1])
    return pred

def compute_bench(path, n_iter=3):

    it = 0

    results = defaultdict(lambda: [])

    for i in xrange(n_iter):
        train, test, test_data_points  = load_movielens(path, seed=i)
        mask = (test != 0)

        it += 1
        print('====================')
        print('Iteration %03d of %03d' % (i+1, n_iter))
        print('====================')

        gc.collect()
        print("benchmarking FactorizationImputer: ")
        tstart = time()
        imp = FactorizationImputer(normalizer=BiScaler(), verbose=False)#may need to add some variables here
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['FactorizationImputer'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking Imputer using mean: ")
        tstart = time()
        imp = Imputer(strategy='mean')
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['Mean Imputer'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking Imputer using most frequent value: ")
        tstart = time()
        imp = Imputer(strategy='most_frequent')
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['Most Frequent Value Imputer'].append((time() - tstart, mse))

        #gc.collect()
        #print("benchmarking Imputer using Neural Net: ")
        #tstart = time()
        #imp = Imputer(strategy='median')
        #pred = neural_net(train)
        #pred *= mask
        #mse = mean_squared_error(test, pred)
        #results['Neural Network'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking GraphLab: ")
        tstart = time()
        pred = gpredict(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['GraphLab'].append((time() - tstart, mse))

    return results, train, test

def plot_results(results):
    mse = dict()
    time = dict()
    for k, v in results.iteritems():
        avg_time = []
        avg_mse = []
        for t in v:
            avg_time.append(t[0])
            avg_mse.append(t[1])
        avg_time = np.mean(avg_time)
        avg_mse = np.mean(avg_mse)
        mse[k] = avg_mse
        time[k] = avg_time
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.bar(range(len(mse)), mse.values(), align='center')
    ax2.bar(range(len(time)), time.values(), align='center')
    ax1.set_title("Mean Squared Error Each Model")
    ax2.set_title("Time To Run Each Model")
    plt.xticks(range(len(time)), time.keys())
    plt.show()


if __name__ == '__main__':
   # results, train, test = compute_bench('movielens.csv', n_iter=3)
   # plot_results(results)
    path = 'movielens.csv'
    train, test, testdatapoints = load_movielens(path)
    #print gpredict(train, testdatapoints)
    #train = train.toarray()
    #bis = biscale(train)
    #print my_predict(train, testdatapoints)
    #bis_masked = bis * (train != 0)
    #cbis = csr_matrix(bis_masked)
    print gpredict(train, testdatapoints)
    print my_predict(train, testdatapoints)
    #print bis
