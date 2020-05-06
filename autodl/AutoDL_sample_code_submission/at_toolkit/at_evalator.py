import numpy as np
from sklearn.metrics import roc_auc_score
from functools import reduce
# from tools import log_warning
from at_toolkit import warning as log_warning

def get_valid_columns(solution):
    """Get a list of column indices for which the column has more than one class.
    This is necessary when computing BAC or AUC which involves true positive and
    true negative in the denominator. When some class is missing, these scores
    don't make sense (or you have to add an epsilon to remedy the situation).

    Args:
        solution: array, a matrix of binary entries, of shape
        (num_examples, num_features)
    Returns:
        valid_columns: a list of indices for which the column has more than one
        class.
    """
    num_examples = solution.shape[0]
    col_sum = np.sum(solution, axis=0)
    valid_columns = np.where(1 - np.isclose(col_sum, 0) -
                             np.isclose(col_sum, num_examples))[0]
    return valid_columns


def tiedrank(a):
    ''' Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays.'''
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        oldval = sa[0]
        k0 = 0
        for k in range(1, m):
            if sa[k] != oldval:
                R[k0: k] = sum(R[k0: k]) / (k - k0)
                k0 = k
                oldval = sa[k]
        R[k0: m] = sum(R[k0: m]) / (m - k0)
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S


def mvmean(R, axis=0):
    ''' Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten.'''
    if len(R.shape) == 0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x))[
        1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))


class ATEvaluator(object):
    # Metric used to compute the score of a point on the learning curve
    @staticmethod
    def autodl_auc(solution, prediction, valid_columns_only=True):
        """Compute normarlized Area under ROC curve (AUC).
        Return Gini index = 2*AUC-1 for  binary classification problems.
        Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
        for the predictions. If solution and prediction are not vectors, the AUC
        of the columns of the matrices are computed and averaged (with no weight).
        The same for all classification problems (in fact it treats well only the
        binary and multilabel classification problems). When `valid_columns` is not
        `None`, only use a subset of columns for computing the score.
        """
        if valid_columns_only:
            valid_columns = get_valid_columns(solution)
            if len(valid_columns) < solution.shape[-1]:
                log_warning("Some columns in solution have only one class, " +
                            "ignoring these columns for evaluation.")
            solution = solution[:, valid_columns].copy()
            prediction = prediction[:, valid_columns].copy()
        label_num = solution.shape[1]
        auc = np.empty(label_num)
        for k in range(label_num):
            r_ = tiedrank(prediction[:, k])
            s_ = solution[:, k]
            if sum(s_) == 0: print("WARNING: no positive class example in class {}" \
                                   .format(k + 1))
            npos = sum(s_ == 1)
            nneg = sum(s_ < 1)
            auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
        return 2 * mvmean(auc) - 1

    @staticmethod
    def auc_metric(solution, prediction, task='binary.classification'):
        '''roc_auc_score() in sklearn is fast than code provided by sponsor
        '''
        if solution.sum(axis=0).min() == 0:
            return np.nan
        auc = roc_auc_score(solution, prediction, average='macro')
        return np.mean(auc * 2 - 1)

    @staticmethod
    def skl_auc_macro(solution, prediction):
        valid_score = roc_auc_score(solution, prediction, average='macro')
        return valid_score
