from datasets import Dataset, load_dataset
import config
import io
from PIL import Image
import requests
import os
import torch



def preprocess_function(samples, processor):
    images = samples.get("image_url")
    paths = samples.get("image_path")
    # paths = [os.path.join(config.IMAGE_PATH , p) for p in paths]
    if os.path.exists(paths[0]) :
        print("loading images from local")
        images = [Image.open(p) for p in paths]
    else :
        print("downloading images")
        images = [Image.open(io.BytesIO(requests.get(url).content)) for url in images]

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



def load_dataset_from_hub (processor, split = None) -> Dataset:
    ds = load_dataset("Harshtherocking/indian-fashion-ecommerce")
    if split :
        ds = ds[split]
    ds.map(lambda x : preprocess_function(x,processor), batched=True, remove_columns=ds.column_names)
    print(ds)
    return ds


if __name__ == "__main__" :

    data_files = {
        "train": "data/train_data.json",
        "validation": "data/val_data.json",
        "test": "data/test_data.json"
    }

    dataset = load_dataset("json", data_files=data_files)
    print(dataset)

    dataset.push_to_hub("Harshtherocking/indian-fashion-ecommerce")
