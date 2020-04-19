# Functions performing various data conversions for the ChaLearn AutoML challenge

# Main contributors: Arthur Pesah and Isabelle Guyon, August-October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import numpy as np
from scipy.sparse import *
from sklearn.datasets import load_svmlight_file
import os 
# Note: to check for nan values np.any(map(np.isnan,X_train))
def file_to_array (filename, verbose=False):
    ''' Converts a file to a list of list of STRING
    It differs from np.genfromtxt in that the number of columns doesn't need to be constant'''
    data =[]
    with open(filename, "r") as data_file:
        if verbose: print ("Reading {}...".format(filename))
        lines = data_file.readlines()
        if verbose: print ("Converting {} to correct array...".format(filename))
        data = [lines[i].strip().split() for i in range (len(lines))]
    return data

def file_to_libsvm (filename, data_binary  , n_features):
    ''' Converts a file to svmlib format and return csr matrix 
    filname = path of file 
    data_binary = True if is sparse binary data False else 
    n_features = number of features
    '''
    data =[]
    with open(filename, "r") as data_file:
        lines = data_file.readlines()
        with open('tmp.txt', 'w') as f:
            for l in lines  :
                tmp = l.strip().split()
                f.write("0 ")
                for i in range (len(tmp) ):
                    if(data_binary):
                        f.write(tmp[i]+":1 ")
                    else:
                        f.write(tmp[i]+" ")
                f.write("\n")
    print ("-------------------- file_to_libsvm  ---------------------")
    l = load_svmlight_file('tmp.txt', zero_based= False ,  n_features = n_features )
    os.remove("tmp.txt")
    return l[0]

def read_first_line (filename):
	''' Read fist line of file'''
	data =[]
	with open(filename, "r") as data_file:
		line = data_file.readline()
		data = line.strip().split()
	return data  
 
def num_lines (filename):
	''' Count the number of lines of file'''
	return sum(1 for line in open(filename))

def binarization (array):
	''' Takes a binary-class datafile and turn the max value (positive class) into 1 and the min into 0'''
	array = np.array(array, dtype=float) # conversion needed to use np.inf after
	if len(np.unique(array)) > 2:
		raise ValueError ("The argument must be a binary-class datafile. {} classes detected".format(len(np.unique(array))))
	
	# manipulation which aims at avoid error in data with for example classes '1' and '2'.
	array[array == np.amax(array)] = np.inf
	array[array == np.amin(array)] = 0
	array[array == np.inf] = 1
	return np.array(array, dtype=int)


def multilabel_to_multiclass (array):
	array = binarization (array)
	return np.array([np.nonzero(array[i,:])[0][0] for i in range (len(array))])
	
def convert_to_num(Ybin, verbose=True):
	''' Convert binary targets to numeric vector (typically classification target values)'''
	if verbose: print("\tConverting to numeric vector")
	Ybin = np.array(Ybin)
	if len(Ybin.shape) ==1:
         return Ybin
	classid=range(Ybin.shape[1])
	Ycont = np.dot(Ybin, classid)
	if verbose: print(Ycont)
	return Ycont
 
def convert_to_bin(Ycont, nval, verbose=True):
    ''' Convert numeric vector to binary (typically classification target values)'''
    if verbose: print ("\t_______ Converting to binary representation")
    Ybin=[[0]*nval for x in xrange(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])]=1
        Ybin[i] = line
    return Ybin


def tp_filter(X, Y, feat_num=1000, verbose=True):
    ''' TP feature selection in the spirit of the winners of the KDD cup 2001
    Only for binary classification and sparse matrices'''
        
    if issparse(X) and len(Y.shape)==1 and len(set(Y))==2 and (sum(Y)/Y.shape[0])<0.1: 
        if verbose: print("========= Filtering features...")
        Posidx=Y>0
        nz=X.nonzero()
        mx=X[nz].max()
        if X[nz].min()==mx: # sparse binary
            if mx!=1: X[nz]=1
            tp=csr_matrix.sum(X[Posidx,:], axis=0)
     
        else:
            tp=np.sum(X[Posidx,:]>0, axis=0)
  

        tp=np.ravel(tp)
        idx=sorted(range(len(tp)), key=tp.__getitem__, reverse=True)   
        return idx[0:feat_num]
    else:
        feat_num = X.shape[1]
        return range(feat_num)
    
def replace_missing(X):
    # This is ugly, but
    try:
        if X.getformat()=='csr':
            return X
    except:
        XX = np.nan_to_num(X)
        
    return XX
