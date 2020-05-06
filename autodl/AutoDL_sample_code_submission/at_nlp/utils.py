import numpy as np

from sklearn.multiclass import OneVsRestClassifier

def color_msg(msg, color="red"):
    if color == "red":
        return '\033[31m{}\033[0m'.format(msg)

    elif color == "blue":
        return '\033[34m{}\033[0m'.format(msg)

    elif color == "yellow":
        return '\033[33m{}\033[0m'.format(msg)

    elif color == "green":
        return '\033[36m{}\033[0m'.format(msg)


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)



