import os

from clearml import Task,Dataset
import hydra
from omegaconf import OmegaConf
import pandas as pd 
import numpy as np

@hydra.main(config_path=os.path.join("..", "..", "config"), config_name="ETL_config")

def inference_main(cfg):
    PROJECT_NAME = "incubation c4"
    TASK_NAME = "maritime_inference_dataset"

    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME,output_uri='s3://experiment-logging')
    task.set_base_docker(
        docker_image="nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04",
    )

    print("Detected config file, initiating task... {}".format(cfg))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)

    if not cfg.local:
        from .check_parent_dataset import create_dataset
        task.execute_remotely(queue_name='compute')

        dataset_dict = Dataset.list_datasets(
            dataset_project=cfg.remote_dataset_project, partial_name=cfg.remote_dataset_name, only_completed=False
        )

        datasets_obj = [
            Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
        ]

        # reverse list due to child-parent dependency, and get the first dataset_obj
        dataset_obj = datasets_obj[::-1][0]
        
        source_folder = dataset_obj.get_local_copy()

        source_file = [file for file in dataset_obj.list_files() if file==cfg.remote_dataset_file_name][0]

        source_path = source_folder + "/" + source_file

        if source_path.endswith('.csv'):
            df = pd.read_csv(source_path) 
        elif source_path.endswith('.parquet'):
            df = pd.read_parquet(source_path,engine='fastparquet')
        elif source_path.endswith('.json'):
            df = pd.read_json(source_path)

    else:
        from check_parent_dataset import create_dataset
        
        if cfg.local_file.endswith('.csv'):
            df = pd.read_csv(cfg.local_file) 
        elif cfg.local_file.endswith('.parquet'):
            df = pd.read_parquet(cfg.local_file,engine='fastparquet')
        elif cfg.local_file.endswith('.json'):
            df = pd.read_json(cfg.local_file)


    inference_df = pd.DataFrame()
    inference_df[cfg.source] = df[cfg.text_field]
    if cfg.rel is None:
        inference_df['doc_id'] = inference_df.index
    else:
        inference_df[cfg.rel] = df[cfg.rel]

    dataset = create_dataset(
        dataset_project=cfg.dataset_project,
        dataset_name=cfg.dataset_name,
    )

    inference_df.to_parquet("test.parquet", engine='fastparquet')
    dataset.add_files("test.parquet")
    dataset.upload()
    dataset.finalize()

if __name__ == "__main__":
    inference_main()