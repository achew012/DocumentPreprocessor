from tempfile import gettempdir
import os

import pandas as pd 
import numpy as np

from datasets import load_dataset, load_from_disk, concatenate_datasets

def merge_to_triples(clean:pd.DataFrame(),unclean:pd.DataFrame(),cfg):
    result = pd.merge(clean,unclean,how="inner",left_on=cfg.rel,right_on=cfg.rel,suffixes=("", "_y"))
    
    if cfg.rel is None:
        result['doc_id'] = result.index

    try:
        if cfg.rename:
            result.rename(columns=cfg.rename, inplace=True)
    except:
        print("Please check that the rename dictionary passed in matched ")

    result.drop(result.columns.difference([cfg.source,cfg.target,cfg.rel]), 1, inplace=True)

    print(result.info())

    return result

def train_validate_test_split(result,train_split,validation_split):
    return np.split(result.sample(frac=1, random_state=42), [int(train_split*len(result)), int((1.0-validation_split)*len(result))])

def parquet_and_upload(dataset,dataframe,parquet_name):
    dataframe.to_parquet(os.path.join(gettempdir(), parquet_name), engine='fastparquet')
    dataset.add_files(os.path.join(gettempdir(), parquet_name))

def dataset_to_shard(dataset, shard_path="/tmp/dataset", num_shards=8):
    for index in range(num_shards):
        dataset.shard(num_shards=num_shards, index=index).save_to_disk(
            "{}/shard_{}".format(shard_path, index)
        )
    return shard_path


def shard_to_dataset(shard_path="/tmp/dataset", num_shards=8):
    dataset = concatenate_datasets(
        [load_from_disk("{}/shard_{}".format(shard_path, i)) for i in range(num_shards)]
    )
    return dataset