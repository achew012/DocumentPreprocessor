from tempfile import gettempdir
import os

from clearml import Task, StorageManager, Logger, Dataset

import pandas as pd 
import numpy as np

from check_parent_dataset import create_dataset

def merge_clean_unclean(clean:pd.DataFrame(),unclean:pd.DataFrame()):
    result = pd.merge(clean,unclean,how="inner",left_on='url',right_on='url',suffixes=("", "_y"))
    try:
        result = result.drop(['timestamp_y'], axis = 1)
        result.rename(columns={'text': 'clean', 'text_y': 'raw'}, inplace=True)
    except:
        print("Please check that for both cleaned and uncleaned dataframes, the column containing the textual data is named as 'text'")
    result['doc_id'] = result.index
    print(result.info())

    return result

def train_validate_test_split(result):
    return np.split(result.sample(frac=1, random_state=42), [int(.6*len(result)), int(.8*len(result))])

def parquet_and_upload(dataset,dataframe,parquet_name):
    dataframe.to_parquet(os.path.join(gettempdir(), parquet_name), engine='fastparquet')
    dataset.add_files(os.path.join(gettempdir(), parquet_name))
