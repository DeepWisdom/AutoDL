# Functions performing various input/output operations for the ChaLearn AutoML challenge

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

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np
import pandas as pd
import os
import shutil
from scipy.sparse import * # used in data_binary_sparse
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
import data_converter
from sys import stderr
from sys import version
from glob import glob as ls
from os import getcwd as pwd
from os.path import isfile

# get_installed_distributions has gone from pip v10
try:
    from pip._internal.utils.misc import get_installed_distributions as lib
except ImportError:  # pip < 10
    from pip import get_installed_distributions as lib

import yaml
from shutil import copy2
import csv
import psutil
import platform

# ================ Small auxiliary functions =================

def read_as_df(basename, type="train"):
    ''' Function to read the AutoML format and return a Panda Data Frame '''
    csvfile = basename + '_' + type + '.csv'
    if isfile(csvfile):
        print('Reading '+ basename + '_' + type + ' from CSV')
        XY = pd.read_csv(csvfile)
        return XY

    print('Reading '+ basename + '_' + type+ ' from AutoML format')
    feat_name = pd.read_csv(basename + '_feat.name', header=None)
    label_name = pd.read_csv(basename + '_label.name', header=None)
    X = pd.read_csv(basename + '_' + type + '.data', sep=' ', names = np.ravel(feat_name))
    [patnum, featnum] = X.shape
    print('Number of examples = %d' % patnum)
    print('Number of features = %d' % featnum)

    XY=X
    Y=[]
    solution_file = basename + '_' + type + '.solution'
    if isfile(solution_file):
        Y = pd.read_csv(solution_file, sep=' ', names = np.ravel(label_name))
        [patnum2, classnum] = Y.shape
        assert(patnum==patnum2)
        print('Number of classes = %d' % classnum)
        # Here we add the target values as a last column, this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        label_range = np.arange(classnum).transpose()         # This is just a column vector [[0], [1], [2]]
        numerical_target = Y.dot(label_range)                 # This is a column vector of dim patnum with numerical categories
        nominal_target = pd.Series(np.array(label_name)[numerical_target].ravel()) # Same with nominal categories
        XY = X.assign(target=nominal_target.values)          # Add the last column

    return XY

# ================ Small auxiliary functions =================

swrite = stderr.write

if (os.name == "nt"):
       filesep = '\\'
else:
       filesep = '/'

def write_list(lst):
    ''' Write a list of items to stderr (for debug purposes)'''
    for item in lst:
        swrite(item + "\n")

def print_dict(verbose, dct):
    ''' Write a dict to stderr (for debug purposes)'''
    if verbose:
        for item in dct:
            print(item + " = " + str(dct[item]))

def mkdir(d):
    ''' Create a new directory'''
    if not os.path.exists(d):
        os.makedirs(d)

def mvdir(source, dest):
    ''' Move a directory'''
    if os.path.exists(source):
        os.rename(source, dest)

def rmdir(d):
    ''' Remove an existingdirectory'''
    if os.path.exists(d):
        shutil.rmtree(d)

def vprint(mode, t):
    ''' Print to stdout, only if in verbose mode'''
    if(mode):
            print(t)

# ================ Output prediction results and prepare code submission =================

def write(filename, predictions):
    ''' Write prediction scores in prescribed format'''
    filename_temp = 'temp_prediction_file_' + str(np.random.randint(10000))
    filename_temp = os.path.join(os.path.dirname(filename), filename_temp)
    with open(filename_temp, "w") as output_file:
        for row in predictions:
            if type(row) is not np.ndarray and type(row) is not list:
                row = [row]
            output_file.write(' '.join(['{0:g}'.format(float(val)) for val in row]))
            output_file.write('\n')
    os.rename(filename_temp, filename)

def zipdir(archivename, basedir):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                if not fn.endswith('.zip'):
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir):] #XXX: relative path
                    assert absfn[:len(basedir)] == basedir
                    if zfn[0] == os.sep:
                        zfn = zfn[1:]
                    z.write(absfn, zfn)

# ================ Inventory input data and create data structure =================

def inventory_data(input_dir):
    ''' Inventory the datasets in the input directory and return them in alphabetical order'''
    # Assume first that there is a hierarchy dataname/dataname_train.data
    training_names = ls(os.path.join(input_dir, '*.data'))
    training_names = [ name.split('/')[-1] for name in training_names ]

    ntr=len(training_names)
    if ntr==0:
        print('WARNING: Inventory data - No data file found')
        training_names = []
    training_names.sort()
    # check_dataset
    return training_names


def check_dataset(dirname, name):
    ''' Check the test and valid files are in the directory, as well as the solution'''
    valid_file = os.path.join(dirname, name + '_valid.data')
    if not os.path.isfile(valid_file):
        print('No validation file for ' + name)
        exit(1)
    test_file = os.path.join(dirname, name + '_test.data')
    if not os.path.isfile(test_file):
        print('No test file for ' + name)
        exit(1)
    # Check the training labels are there
    training_solution = os.path.join(dirname, name + '_train.solution')
    if not os.path.isfile(training_solution):
        print('No training labels for ' + name)
        exit(1)
    return True


def data(filename, nbr_features=None, verbose = False):
    ''' The 2nd parameter makes possible a using of the 3 functions of data reading (data, data_sparse, data_binary_sparse) without changing parameters'''
    if verbose: print (np.array(data_converter.file_to_array(filename)))
    return np.array(data_converter.file_to_array(filename), dtype=float)

def data_sparse (filename, nbr_features):
    ''' This function takes as argument a file representing a sparse matrix
    sparse_matrix[i][j] = "a:b" means matrix[i][a] = basename and load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm (filename = filename, data_binary = False  , n_features = nbr_features)



def data_binary_sparse (filename , nbr_features):
    ''' This fuction takes as argument a file representing a sparse binary matrix
    sparse_binary_matrix[i][j] = "a"and transforms it temporarily into file svmlibs format( <index2>:<value2>)
    to load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm (filename = filename, data_binary = True  , n_features = nbr_features)



# ================ Copy results from input to output ==========================

def copy_results(datanames, result_dir, output_dir, verbose):
    ''' This function copies all the [dataname.predict] results from result_dir to output_dir'''
    missing_files = []
    for basename in datanames:
        try:
            missing = False
            test_files = ls(result_dir + "/" + basename + "*_test*.predict")
            if len(test_files)==0:
                vprint(verbose, "[-] Missing 'test' result files for " + basename)
                missing = True
            valid_files = ls(result_dir + "/" + basename + "*_valid*.predict")
            if len(valid_files)==0:
                vprint(verbose, "[-] Missing 'valid' result files for " + basename)
                missing = True
            if missing == False:
                for f in test_files: copy2(f, output_dir)
                for f in valid_files: copy2(f, output_dir)
                vprint( verbose,  "[+] " + basename.capitalize() + " copied")
            else:
                missing_files.append(basename)
        except:
            vprint(verbose, "[-] Missing result files")
            return datanames
    return missing_files

# ================ Display directory structure and code version (for debug purposes) =================

def show_dir(run_dir):
    print('\n=== Listing run dir ===')
    write_list(ls(run_dir))
    write_list(ls(run_dir + '/*'))
    write_list(ls(run_dir + '/*/*'))
    write_list(ls(run_dir + '/*/*/*'))
    write_list(ls(run_dir + '/*/*/*/*'))

def show_io(input_dir, output_dir):
    swrite('\n=== DIRECTORIES ===\n\n')
    # Show this directory
    swrite("-- Current directory " + pwd() + ":\n")
    write_list(ls('.'))
    write_list(ls('./*'))
    write_list(ls('./*/*'))
    swrite("\n")

    # List input and output directories
    swrite("-- Input directory " + input_dir + ":\n")
    write_list(ls(input_dir))
    write_list(ls(input_dir + '/*'))
    write_list(ls(input_dir + '/*/*'))
    write_list(ls(input_dir + '/*/*/*'))
    swrite("\n")
    swrite("-- Output directory  " + output_dir + ":\n")
    write_list(ls(output_dir))
    write_list(ls(output_dir + '/*'))
    swrite("\n")

    # write meta data to sdterr
    swrite('\n=== METADATA ===\n\n')
    swrite("-- Current directory " + pwd() + ":\n")
    try:
        metadata = yaml.load(open('metadata', 'r'))
        for key,value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
    except:
        swrite("none\n");
    swrite("-- Input directory " + input_dir + ":\n")
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
        for key,value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
        swrite("\n")
    except:
        swrite("none\n");

def show_version():
    # Python version and library versions
    swrite('\n=== VERSIONS ===\n\n')
    # Python version
    swrite("Python version: " + version + "\n\n")
    # Give information on the version installed
    swrite("Versions of libraries installed:\n")
    map(swrite, sorted(["%s==%s\n" % (i.key, i.version) for i in lib()]))

 # Compute the total memory size of an object in bytes

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

    # write the results in a csv file
def platform_score ( basename , mem_used ,n_estimators , time_spent , time_budget ):
# write the results and platform information in a csv file (performance.csv)
    with open('performance.csv', 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        #['Data name','Nb estimators','System', 'Machine' , 'Platform' ,'memory used (Mb)' , 'number of CPU' ,' time spent (sec)' , 'time budget (sec)'],
        data = [
        [basename,n_estimators,platform.system(), platform.machine(),platform.platform() , float("{0:.2f}".format(mem_used/1048576.0)) , str(psutil.cpu_count()) , float("{0:.2f}".format(time_spent)) ,    time_budget ]
        ]
        a.writerows(data)
