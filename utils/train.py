from peft import PeftModel, get_peft_model
import config
from torch.optim import AdamW, lr_scheduler
from utils.dataset import blip_data_collator
from math import ceil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


DEVICE = config.DEVICE


def prepare_lora_model (model, lora_config = config.lora_config) -> PeftModel:
    model.to(DEVICE)
    peft_model = get_peft_model(model, lora_config)
    print(peft_model.print_trainable_parameters())
    return peft_model


def train (model, dataset) :
    peft_model = prepare_lora_model(model)
    peft_model.to(DEVICE)
    model.train()

    optimizer  = AdamW(model.parameters(), lr= config.LR)
    optimizer.zero_grad()

    # scheduler = lr_scheduler.LRScheduler(optimizer=optimizer)

    writer = SummaryWriter()
    global_step = 0

    for epoch in range(config.EPOCHS) :
        ds_size = len(dataset)
        num_batches = ceil(ds_size /config.BATCH_SIZE)
        print(f"Number of batches: {num_batches}")

        epoch_loss = 0
        for batch in tqdm(range(num_batches),desc=f"Epoch {epoch}") :
            end = min( (batch+1) * config.BATCH_SIZE, ds_size)
            batch_data = dataset[batch * config.BATCH_SIZE:end]

            batch_data = blip_data_collator(batch_data)
            labels = batch_data["labels"].to(DEVICE)
            pixel_values = batch_data["pixel_values"].to(DEVICE)
            attention_mask = batch_data["attention_mask"].to(DEVICE)
            input_ids = batch_data["input_ids"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels,
                            attention_mask=attention_mask)

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        # scheduler.step()
        avg_epoch_loss = epoch_loss / ds_size
        print(f"Epoch {epoch} loss: {avg_epoch_loss}")
        writer.add_scalar("Loss/epoch_avg", avg_epoch_loss, epoch)

    print("Finished Training")
    writer.close()
    model.save_pretrained(config.OUTPUT_DIR, safe_serialization=False)
    print(f"Saved model to {config.OUTPUT_DIR}")
