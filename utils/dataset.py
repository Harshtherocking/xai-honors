from datasets import Dataset, load_dataset

def load_dataset_from_hub () -> Dataset:
    ds = load_dataset("Harshtherocking/indian-fashion-ecommerce")
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
