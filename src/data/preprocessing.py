'''
PROCESSING STRATEGY:
0. Read Parquet file
1. Take diff of dirty tokens and clean tokens as dirty vocab ###
2. Randomly swap clean tokens with dirty tokens from dirty vocab and mask it ###
3. Masking strategy 0.2 chance of original, 0.6 chance of dirty, 0.1 chance of empy token, 0.1 chance random token ###
4. Return a Dataframe
'''

from transformers.models.led import LEDConfig, LEDTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from clearml import Dataset
import ipdb


def get_dataset():
    clearml_dataset_project_name = "datasets/c4"
    clearml_dataset_name = "c4_raw_clean"
    clearml_data_object = Dataset.get(
        dataset_name=clearml_dataset_name,
        dataset_project=clearml_dataset_project_name,
    )
    dataset_path = clearml_data_object.get_local_copy()
    dataset = pd.read_parquet(dataset_path, engine="fastparquet")[:10]
    return dataset


# def get_raw_vocab(raw: pd.Series, clean: pd.Series, tokenizer: object) -> pd.Series:
#     # raw_vocab = set(tokenizer.tokenize(
#     #     raw.iloc[0])).difference(set(tokenizer.tokenize(clean.iloc[0])))
#     return

def mask_clean_tokens(dataset: pd.Series, tokenizer: object, max_mask_span: int = 10, masking_strategy_list: list = ["dirty", "clean", "empty"]) -> torch.Tensor:
    clean_ids = tokenizer(dataset["clean"].tolist())["input_ids"]
    dirty_ids = tokenizer(dataset["raw"].tolist())["input_ids"]

    new_source_ids = []
    new_target_ids = []

    for clean_id, dirty_id in tqdm(zip(clean_ids, dirty_ids)):
        masking_strategy = np.random.choice(
            masking_strategy_list, 1, replace=True, p=[0.1, 0.6, 0.3])[0]

        # Get random span from clean text
        clean_start_idx = np.random.randint(0, len(clean_id)-max_mask_span)
        span_offset = np.random.randint(1, max_mask_span)
        clean_end_idx = clean_start_idx + span_offset
        masked_clean_id = clean_id

        if masking_strategy == "dirty":
            # Sample from dirty and add noise to the clean text
            dirty_start_idx = np.random.randint(
                0, len(dirty_id)-max_mask_span)
            dirty_end_idx = dirty_start_idx + span_offset
            clean_id[clean_start_idx:clean_end_idx] = dirty_id[dirty_start_idx:dirty_end_idx]
            masked_clean_id[clean_start_idx:clean_end_idx] = [
                tokenizer.mask_token_id]

            new_source_ids.append(tokenizer.decode(masked_clean_id[1:-2]))
            new_target_ids.append(tokenizer.decode(clean_id[1:-2]))

        elif masking_strategy == "clean":
            # Mask the clean span
            masked_clean_id[clean_start_idx:clean_end_idx] = [
                tokenizer.mask_token_id]

            new_source_ids.append(tokenizer.decode(masked_clean_id[1:-2]))
            new_target_ids.append(tokenizer.decode(clean_id[1:-2]))

        elif masking_strategy == "empty":
            # Remove dirty spans from dirty text
            masked_dirty_id = dirty_id
            dirty_start_idx = np.random.randint(
                0, len(dirty_id)-max_mask_span)
            dirty_end_idx = dirty_start_idx + span_offset
            # dirty_id = dirty_id[:dirty_start_idx] + dirty_id[dirty_end_idx:]

            masked_dirty_id[dirty_start_idx:dirty_end_idx] = [
                tokenizer.mask_token_id]

            new_source_ids.append(tokenizer.decode(masked_dirty_id[1:-2]))
            new_target_ids.append(tokenizer.decode(clean_id[1:-2]))

        #entities_samples = random.choices(doc["entities"], k=self.max_spans)
        #entities_samples = random.sample(doc["entities"], k=self.max_spans)

    return new_source_ids, new_target_ids


def data_formatting(dataset: pd.DataFrame, tokenizer: object) -> torch.Tensor:
    # raw = dataset["raw"]
    masked_tokens, original_tokens = mask_clean_tokens(dataset, tokenizer)

    masked_tokens_ids = tokenizer(masked_tokens, padding="max_length",
                                  max_length=1024, truncation=True, return_tensors="pt")
    tokens_ids = tokenizer(original_tokens, padding="max_length",
                           max_length=1024, truncation=True, return_tensors="pt")

    return masked_tokens_ids, tokens_ids


tokenizer = LEDTokenizer.from_pretrained(
    'allenai/led-base-16384', use_fast=True)
data_formatting(get_dataset(), tokenizer)
