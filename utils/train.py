import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer
import config


DEVICE = config.DEVICE


def prepare_lora_model (model, lora_config = config.lora_config) :
    model.to(DEVICE)
    peft_model = get_peft_model(model, lora_config)
    print(peft_model.print_trainable_parameters())
    return peft_model


def train (model, dataset) :
    peft_model = prepare_lora_model(model)
    trainer = Trainer(
        model=peft_model,
        args=config.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
    )
    trainer.train()
    trainer.save_model(config.OUTPUT_DIR)