import numpy as np
import pandas as pd
from Auto_Tabular import CONSTANT


def ohe2cat(label):
    return np.argmax(label, axis=1)


def downcast(series, accuracy_loss=True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0
            
            if np.isnan(min_value):
                min_value = 0
                
            if min_float_type=='float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if(len(tmp)==0):
                return series.astype(np.float16)
            
            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)
           
            else:
                return series
            
    else:
        return series


def check_density(sparse_matrix, num_feat=100):
    density = np.array(sparse_matrix.sum(axis=0)/sparse_matrix.shape[0])[0]
    sort_index = sorted(range(len(density)), key=lambda k: density[k], reverse=True)
    return sort_index[:min(num_feat,len(density))]


def fill_na(df):
    columns = df.columns
    for col in columns:
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            df[col].fillna(df[col].max()+1, inplace=True)
        elif col.startswith(CONSTANT.NUMERICAL_PREFIX):
            df[col].fillna(df[col].astype(float).mean(), inplace=True)
    return df



