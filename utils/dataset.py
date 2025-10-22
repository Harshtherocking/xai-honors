from datasets import Dataset, load_dataset, DatasetDict
import config
import io
from PIL import Image
import requests
import os
import numpy as np
import torch


num_proc = os.cpu_count()

def preprocess_function(samples, processor):
    images_url = samples.get("image_url")
    paths = samples.get("image_path")
    # paths = [os.path.join(config.IMAGE_PATH , p) for p in paths]

    images = []
    for p, url in zip(paths,images_url):
        if os.path.exists(p) :
            images.append(Image.open(p))
        else :
            try :
                images.append(Image.open(io.BytesIO(requests.get(url).content)))
            except Exception as e:
                print(f"Exception : {e}\nFile : {p}\n --- using black image")
                images.append(np.zeros(shape=(244,244,3)))


    inputs = processor(images=images, return_tensors="pt")  # pixel_values present

    captions = samples.get("product_title")

    tokenized = processor.tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_TARGET_LENGTH,
        return_tensors="pt",
    )
    labels = tokenized["input_ids"]
    # pad tokens to -100 to ignore in cross entropy loss
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch = {}
    batch["pixel_values"] = inputs.get('pixel_values')
    batch["labels"] = labels
    batch["attention_mask"] = tokenized["attention_mask"]
    return batch



def load_dataset_from_hub (processor, split = None, subset_size= None, num_proc = num_proc) -> Dataset:
    dataset = load_dataset("Harshtherocking/indian-fashion-ecommerce")

    # If a specific split is provided
    if split is not None:
        ds_split = dataset[split]

        # subset if needed
        if subset_size is not None and subset_size < len(ds_split):
            ds_split = ds_split.shuffle(seed=42).select(range(subset_size))

        ds_split = ds_split.map(
            lambda x: preprocess_function(x, processor),
            batched=True,
            num_proc=num_proc,
            remove_columns=ds_split.column_names,
            load_from_cache_file=True,
            desc=f"Processing {split} split",
        )
        return ds_split  # single split case → returns Dataset

    # Else process all splits
    processed_splits = {}
    for split_name, ds_split in dataset.items():
        if subset_size is not None and subset_size < len(ds_split):
            ds_split = ds_split.shuffle(seed=42).select(range(subset_size))

        ds_split = ds_split.map(
            lambda x: preprocess_function(x, processor),
            batched=True,
            num_proc=num_proc,
            remove_columns=ds_split.column_names,
            load_from_cache_file=True,
            desc=f"Processing {split_name} split",
        )
        processed_splits[split_name] = ds_split

    processed_dataset = DatasetDict(processed_splits)
    return processed_dataset

    # processed_splits = {}
    # for split, ds_split in dataset.items():
    #     # take subset if specified
    #     if subset_size is not None and subset_size < len(ds_split):
    #         ds_split = ds_split.shuffle(seed=42).select(range(subset_size))
    #
    #     # parallel preprocessing
    #     ds_split = ds_split.map(
    #         lambda x: preprocess_function(x, processor),
    #         batched=True,
    #         num_proc=num_proc,
    #         remove_columns=ds_split.column_names,
    #         load_from_cache_file=True,
    #         desc=f"Processing {split} split",
    #     )
    #
    #     processed_splits[split] = ds_split
    #
    # # convert to DatasetDict (Hugging Face object)
    # processed_dataset = DatasetDict(processed_splits)
    # return processed_dataset

    # if subset_size:
    #     if split :
    #         ds = ds.shuffle(seed=42).select(range(subset_size))
    #     else :
    #         for sp in ds.keys() :
    #             ds = {k:v for k,v in zip(ds.keys(), ds[])}
    #             pass
    #
    # if split :
    #     ds = ds[split]
    #
    # print(ds)
    #
    # ds.map(lambda x : preprocess_function(x,processor),
    #        batched=True,
    #        remove_columns=ds.column_names,
    #        num_proc= num_proc,
    #        load_from_cache_file=True)
    # print(ds)
    # return ds


if __name__ == "__main__" :

    data_files = {
        "train": "data/train_data.json",
        "validation": "data/val_data.json",
        "test": "data/test_data.json"
    }

    dataset = load_dataset("json", data_files=data_files)
    print(dataset)

    dataset.push_to_hub("Harshtherocking/indian-fashion-ecommerce")
