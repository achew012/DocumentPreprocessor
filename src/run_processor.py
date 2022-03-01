import hydra
from omegaconf import OmegaConf
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from clearml import Task, StorageManager, Dataset as ClearML_Dataset
from torch.utils.data import DataLoader

from transformers.models.led import LEDTokenizer
from data.data import PreprocessingDataset
from models.model import LongformerDenoiser

Task.force_requirements_env_freeze(force=True, requirements_file="requirements.txt")
Task.add_requirements("rouge_score")
Task.add_requirements("nltk")


def get_dataloader(split_name, cfg):
    """Get training and validation dataloaders"""
    # train_data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
    # self.c4_train = load_dataset(
    #     "allenai/c4", data_files=train_data_files, split="train")
    # val_data_files = {"validation": "en/c4-validation.*.json.gz"}
    # self.c4_validation = load_dataset(
    #     "allenai/c4", data_files=val_data_files, split="validation[:10%]")
    # self.c4_test = load_dataset(
    #     "allenai/c4", data_files=val_data_files, split="validation[10%:20%]")

    # if cfg.clearml_dataset_project_name and cfg.clearml_dataset_name:
    clearml_data_object = ClearML_Dataset.get(
        dataset_name=cfg.clearml_dataset_name,
        dataset_project=cfg.clearml_dataset_project_name,
        dataset_tags=list(cfg.clearml_dataset_tags),
        # only_published=True,
    )
    dataset_path = clearml_data_object.get_local_copy()

    if cfg.debug:
        dataset_split = dataset_split[:10]

    tokenizer = LEDTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    dataset = PreprocessingDataset(
        dataset_path=os.path.join(
            dataset_path,
            "{}.parquet".format(split_name),
        ),
        tokenizer=tokenizer,
        cfg=cfg,
    )

    if split_name in ["validate", "test"]:
        return DataLoader(
            dataset,
            batch_size=cfg.eval_batch_size,
            num_workers=cfg.num_workers,
            collate_fn=PreprocessingDataset.collate_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=PreprocessingDataset.collate_fn,
        )


def train(cfg, task) -> LongformerDenoiser:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./",
        filename="best_model",
        monitor="average_val_rouge1",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
        period=5,
    )

    train_loader = get_dataloader("train", cfg)
    val_loader = get_dataloader("validate", cfg)

    model = LongformerDenoiser(cfg, task)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    return model


def test(cfg, model) -> list:
    test_loader = get_dataloader("test", cfg)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs)
    results = trainer.test(model, test_loader)
    return results


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.train:
        task = Task.init(
            project_name="DocumentProcessing",
            task_name="LED-Denoiser-train",
            output_uri="s3://experiment-logging/storage/",
        )
    else:
        task = Task.init(
            project_name="DocumentProcessing",
            task_name="LED-Denoiser-test",
            output_uri="s3://experiment-logging/storage/",
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    # task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
    # task.execute_remotely(queue_name="compute", exit_process=True)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        if cfg.trained_model_path:
            trained_model_path = StorageManager.get_local_copy(cfg.trained_model_path)
            model = LongformerDenoiser.load_from_checkpoint(
                trained_model_path, cfg=cfg, task=task
            )

        results = test(cfg, model)


if __name__ == "__main__":
    hydra_main()
