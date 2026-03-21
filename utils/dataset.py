from datasets import Dataset, load_dataset, DatasetDict
import config
import io
from PIL import Image
import requests
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

num_proc = os.cpu_count()

def preprocess_function(samples, processor):
    images_url = samples.get("image_url")
    paths = samples.get("image_path")
    # paths = [os.path.join(config.IMAGE_PATH , p) for p in paths]

    images = []
    # for p, url in zip(paths,images_url):
    #     if os.path.exists(p) :
    #         images.append(Image.open(p))
    #     else :
    #         try :
    #             images.append(Image.open(io.BytesIO(requests.get(url).content)))
    #         except Exception as e:
    #             print(f"Exception : {e}\nFile : {p}\n --- using black image")
    #             images.append(np.zeros(shape=(244,244,3)))

    for p, url in zip(paths, images_url):
        if os.path.exists(p):
            with Image.open(p) as img:
                img.load()
                images.append(img.copy())
        else:
            try:
                with Image.open(io.BytesIO(requests.get(url).content)) as img:
                    img.load()
                    images.append(img.copy())
            except Exception as e:
                print(f"Exception : {e}\nFile : {p}\n --- using black image")
                images.append(np.zeros(shape=(244, 244, 3)))

    captions = samples.get("product_title")

    if config.MODEL == "paligemma":
        model_inputs = processor(
            text = "describe this image : ",
            suffix = captions,
            images = images,
            # padding="max_length",
            # truncation=True,
            # max_length=config.MAX_TARGET_LENGTH,
            return_tensors="pt",

        )
        return model_inputs

    inputs = processor(images=images, return_tensors="pt")  # pixel_values present

    if config.MODEL == "blip" :
        input_tokens  = processor.tokenizer(
            text = "a photograph of",
            padding = "max_length",
            truncation = True,
            max_length = config.MAX_TARGET_LENGTH,
            return_tensors = "pt"
        )
        tokenized = processor.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_TARGET_LENGTH,
            return_tensors="pt",
        )

        input_ids = input_tokens["input_ids"]
        labels = tokenized["input_ids"].clone()
        # pad tokens to -100 to ignore in cross entropy loss
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch = {"pixel_values": inputs.get('pixel_values'), "labels": labels,
                 "attention_mask": input_tokens["attention_mask"], "input_ids": input_ids}
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



# def data_collator(features):
#     batch = {}
#     for key in features[0].keys():
#         batch[key] = torch.tensor([f[key] for f in features])
#     return batch

    # pixel_values = torch.tensor(batch["pixel_values"])
    # labels = torch.tensor(batch["labels"])
    # attention_mask = torch.tensor(batch["attention_mask"])
    # input_ids = torch.tensor(batch["input_ids"])
    #
    # if config.MODEL == "paligemma":
    #     token_type_ids = torch.tensor(batch["token_type_ids"])
    #     return {
    #         "pixel_values": pixel_values,
    #         "labels": labels,
    #         "attention_mask": attention_mask,
    #         "input_ids": input_ids,
    #         "token_type_ids": token_type_ids,
    #     }
    #
    # return {
    #     "pixel_values": pixel_values,
    #     "labels": labels,
    #     "attention_mask": attention_mask,
    #     "input_ids": input_ids,
    # }

def data_collator(features):
    batch = {}

    # pixel_values (already same shape)
    batch["pixel_values"] = torch.stack(
        [torch.tensor(x) for x in features["pixel_values"]]
    )

    # variable-length sequences
    input_ids = [torch.tensor(x) for x in features["input_ids"]]
    attention_mask = [torch.tensor(x) for x in features['attention_mask']]
    labels = [torch.tensor(x) for x in features["labels"]]

    # dynamic padding
    batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=0)
    batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)

    # optional token_type_ids (PaliGemma specific)
    if config.MODEL == "paligemma":
        token_type_ids = [torch.tensor(x) for x in features["token_type_ids"]]
        batch["token_type_ids"] = pad_sequence(
            token_type_ids, batch_first=True, padding_value=0
        )

    return batch
