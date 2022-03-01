from clearml import Task, StorageManager, Logger, Dataset

import pandas as pd 
import numpy as np

from check_parent_dataset import create_dataset

PROJECT_NAME = "incubation c4"
TASK_NAME = "maritime_inference_dataset"
DATASET_PROJECT = "datasets/c4"
DATASET_NAME = "maritime_inference"
LOCAL_FILE = '././data/maritime_data_compiled.csv'

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME,output_uri='s3://experiment-logging')
task.set_base_docker(
    docker_image="nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04",
)

args = {
    "dataset_name": DATASET_NAME,
    "dataset_type": "inference",
    "dataset_split": "train",
    "dataset_project":DATASET_PROJECT
}

task.connect(args)
# task.execute_remotely(queue_name='compute')

print(args)

df = pd.read_csv(LOCAL_FILE) 

inference_df = pd.DataFrame()
inference_df['raw'] = df['texts']
inference_df['doc_id'] = inference_df.index

dataset = create_dataset(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
)

inference_df.to_parquet("test.parquet", engine='fastparquet')
dataset.add_files("test.parquet")
dataset.upload()
dataset.finalize()