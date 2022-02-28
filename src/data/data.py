import torch
from torch.utils.data import Dataset
import pandas as pd
import ipdb


class PreprocessingDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset_path, tokenizer, cfg):
        self.tokenizer = tokenizer

        self.dataset = pd.read_parquet(dataset_path, engine="fastparquet")

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {}
        ipdb.set_trace()
        row = self.dataset
        source = self.tokenizer(row["raw"], return_tensors="pt")
        item["src_input_ids"] = source["input_ids"]
        item["src_attention_mask"] = source["attention_mask"]

        target = self.tokenizer(row["clean"].iloc[0], return_tensors="pt")
        item["tgt_input_ids"] = source["input_ids"]
        item["tgt_attention_mask"] = source["attention_mask"]

        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        src_input_ids = torch.stack([ex["src_input_ids"] for ex in batch])
        src_attention_mask = torch.stack([ex["src_attention_mask"] for ex in batch])
        tgt_input_ids = torch.stack([ex["tgt_input_ids"] for ex in batch])
        tgt_attention_mask = torch.stack([ex["tgt_attention_mask"] for ex in batch])

        return {
            "src_input_ids": src_input_ids,
            "src_attention_mask": src_attention_mask,
            "tgt_input_ids": tgt_input_ids,
            "tgt_attention_mask": tgt_attention_mask,
        }
