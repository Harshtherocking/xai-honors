from peft import PeftModel, get_peft_model
import config
from torch.optim import AdamW, lr_scheduler
from utils.dataset import blip_data_collator
from math import ceil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset

from utils.validate import validate

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration, AutoProcessor
)


def prepare_lora_model (model, lora_config = config.lora_config) -> PeftModel:
    peft_model = get_peft_model(model, lora_config)
    print(peft_model.print_trainable_parameters())
    return peft_model


def train (model, dataset, processor : AutoProcessor, val_dataset : Dataset | None = None) :
    peft_model = prepare_lora_model(model)
    peft_model = peft_model.to(config.DEVICE)

    optimizer  = AdamW(model.parameters(), lr= config.LR)
    writer = SummaryWriter(config.LOG_DIR)
    global_step = 0

    for epoch in range(config.EPOCHS) :
        peft_model.train()
        ds_size = len(dataset)
        num_batches = ceil(ds_size /config.BATCH_SIZE)
        # print(f"Number of batches: {num_batches}")

        epoch_loss = 0
        for batch in tqdm(range(num_batches),desc=f"Epoch {epoch}") :
            end = min( (batch+1) * config.BATCH_SIZE, ds_size)
            batch_data = dataset[batch * config.BATCH_SIZE:end]

            batch_data = blip_data_collator(batch_data)
            labels = batch_data["labels"].to(config.DEVICE)
            pixel_values = batch_data["pixel_values"].to(config.DEVICE)
            attention_mask = batch_data["attention_mask"].to(config.DEVICE)
            input_ids = batch_data["input_ids"].to(config.DEVICE)

            outputs = peft_model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels,
                            attention_mask=attention_mask)

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1


        # validation
        if val_dataset is not None :
            batch_data = blip_data_collator(val_dataset)
            labels = batch_data["labels"].to(config.DEVICE)
            pixel_values = batch_data["pixel_values"].to(config.DEVICE)
            attention_mask = batch_data["attention_mask"].to(config.DEVICE)
            input_ids = batch_data["input_ids"].to(config.DEVICE)
            peft_model.eval()
            outputs = peft_model(input_ids=input_ids,
                                 pixel_values=pixel_values,
                                 labels=labels,
                                 attention_mask=attention_mask)
            bleu, rogue, chrf = validate(outputs, batch_data, processor)

            writer.add_scalar("Validation/BLEU", bleu, epoch)
            writer.add_scalar("Validation/ROGUE", rogue, epoch)
            writer.add_scalar("Validation/CHRF++", chrf, epoch)

    print("Finished Training")
    writer.close()
    peft_model.save_pretrained(config.OUTPUT_DIR, safe_serialization=False)
    print(f"Saved model to {config.OUTPUT_DIR}")

