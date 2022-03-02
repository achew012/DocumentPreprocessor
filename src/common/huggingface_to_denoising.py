import os

from clearml import Task, StorageManager, Logger, Dataset
import pandas as pd 
import numpy as np
import hydra
from omegaconf import OmegaConf


print(os.path.join("..", "..", "config"))
@hydra.main(config_path=os.path.join("..", "..", "config"), config_name="ETL_config")

def huggingface_main(cfg):
    PROJECT_NAME = "incubation c4"
    TASK_NAME = "dataset_store_c4_dataset"
    DATASET_PROJECT = "datasets/c4"
    DATASET_NAME = "c4_test_refractored"

    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME,output_uri='s3://experiment-logging')
    task.set_base_docker(
        docker_image="nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04",
    )

    print("Detected config file, initiating task... {}".format(cfg))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    task.execute_remotely(queue_name='compute')

    from .check_parent_dataset import create_dataset
    from .data_utils import merge_clean_unclean,dataset_to_shard,shard_to_dataset,train_validate_test_split,parquet_and_upload

    clean_dataset_path = cfg.clean_dataset_path
    clean_dataset_name = cfg.clean_dataset_name
    clean_dataset_variant_name = cfg.clean_dataset_variant_name
    clean_dataset_split = cfg.clean_dataset_split
    clean_data_files = cfg.clean_data_files

    unclean_dataset_path = cfg.unclean_dataset_path
    unclean_dataset_name = cfg.unclean_dataset_name
    unclean_dataset_variant_name = cfg.unclean_dataset_variant_name
    unclean_dataset_split = cfg.unclean_dataset_split
    unclean_data_files = cfg.unclean_data_files


    ################## dataset_store_c4_datasets #################
    from datasets import load_dataset

    cleaned_dataset = load_dataset(
        path=clean_dataset_path,name=clean_dataset_variant_name,data_files=clean_data_files,split=clean_dataset_split
    )


    uncleaned_dataset = load_dataset(
        path=unclean_dataset_path,name=unclean_dataset_variant_name,data_files=unclean_data_files,split=unclean_dataset_split
    )

    print("Number of samples in {} dataset".format(cfg.clean_dataset_name), cleaned_dataset.num_rows)
    print("Number of samples in {} dataset".format(cfg.unclean_dataset_name), uncleaned_dataset.num_rows)

    ## shard dataset and save
    cleaned_shard_path = dataset_to_shard(cleaned_dataset,shard_path="/tmp/cleaned_dataset")
    uncleaned_shard_path = dataset_to_shard(uncleaned_dataset,shard_path="/tmp/uncleaned_dataset")

    ## load dataset from shards
    clean_dataset = shard_to_dataset(cleaned_shard_path, num_shards=8)
    unclean_dataset = shard_to_dataset(uncleaned_shard_path, num_shards=8)

    clean_df = clean_dataset.to_pandas()
    print(clean_df.info())
    unclean_df = unclean_dataset.to_pandas()
    print(unclean_df.info())


    result = merge_clean_unclean(clean_df,unclean_df,cfg)

    #split dataset into train, validate, test
    train, validate, test = train_validate_test_split(result,cfg.train_split,cfg.validation_split)

    dataset = create_dataset(
        dataset_project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
    )

    parquet_and_upload(dataset,train,"train.parquet")
    parquet_and_upload(dataset,validate,"validate.parquet")
    parquet_and_upload(dataset,test,"test.parquet")


    dataset.upload()
    dataset.finalize()

if __name__ == "__main__":
    huggingface_main()

