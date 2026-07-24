# imports
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW, lr_scheduler

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer

from tqdm.auto import tqdm
from nano_gpt_model import NanoGPT

import gc
import time

# -----------------------------

# setup cuda
#torch.cuda.memory._record_memory_history()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device("cpu")
print(f"using {device}")
# -----------------------------

# load dataset
dataset = load_dataset("roneneldan/TinyStories")
tokenizer_file = "data/TinyStories-tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_file)
# -----------------------------

# hyperparameters
hyperparameters = {
  "n_epochs": 3,
  "vocab_size": tokenizer.get_vocab_size(),
  "batch_size": 8,
  "block_size": 1080,
  "learning_rate": 1.5e-2,
  "n_embed": 256,
  "n_heads": 2,
  "n_layers": 4,
  "dropout": 0.1,
}

n_epochs = hyperparameters['n_epochs']
vocab_size = hyperparameters['vocab_size']
batch_size = hyperparameters['batch_size']
block_size = hyperparameters['block_size']
learning_rate = hyperparameters['learning_rate']
n_embed = hyperparameters['n_embed']
n_heads = hyperparameters['n_heads']
n_layers = hyperparameters['n_layers']
dropout = hyperparameters['dropout']
# -----------------------------

# tokenize dataset
tokenizer.enable_padding(pad_id=2, pad_token="<|im_end|>", length=block_size)
tokenizer.enable_truncation(max_length=block_size)





tokenized_data = dataset.map(lambda x: { "input_ids": [elem.ids for elem in tokenizer.encode_batch(x['text'])] }, batched=True)
tokenized_data = tokenized_data.with_format("torch")

train_ids = tokenized_data['train'].remove_columns(['text'])
train_ids = train_ids.shuffle().select(range(30000))
train_ids.set_format(type="torch", columns=["input_ids"])
train_ids = train_ids.with_format("torch")
train_ids = train_ids["input_ids"]
train_dataset = TensorDataset(train_ids)

val_ids = tokenized_data['validation'].remove_columns(['text'])
val_ids = val_ids.shuffle().select(range(3000))
val_ids.set_format(type="torch", columns=["input_ids"])
val_ids = val_ids.with_format("torch")
val_ids   = val_ids["input_ids"]
val_dataset = TensorDataset(val_ids)





# -----------------------------

# setup model and trainer


checkpoint = torch.load("3-epoch-4.484M-checkpoint-2.pt")
hyperparameters = checkpoint['hyperparameters']
model = NanoGPT(hyperparameters, device).to(device)
model.load_state_dict(checkpoint['model'])

print(model)
num_params = sum(p.numel() for p in model.parameters())/1e6

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, persistent_workers=False)

print(f"{num_params:.3f}M parameters")
# -----------------------------
            
with torch.no_grad():
    # evaluate validation loss
    model.eval() # switch model to evaluation mode
    losses =  torch.zeros(len(val_dataloader), device=device)
    k = 0
    for (batch,) in val_dataloader:
        batch = batch.to(device)
        targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
        logits, loss = model(batch, targets)
                
        losses[k] = loss.item()
        predictions = torch.argmax(logits, dim=-1)
        k += 1

    avg_val_loss = losses.mean()
    print(f"val loss: {avg_val_loss}")
    # -----------------------------
        
    # evaluate training loss
    losses =  torch.zeros(len(val_dataloader), device=device)
    k = 0
    for (batch,) in train_dataloader:
        batch = batch.to(device)
        targets = torch.concat((batch[:, 1:], 2 * torch.ones([batch.shape[0], 1]).to(device)), dim=1).long()
        logits, loss = model(batch, targets)
                
        losses[k] = loss.item()
        predictions = torch.argmax(logits, dim=-1)
        k += 1
            
        if(k == len(val_dataloader)):
            break
        
    avg_train_loss = losses.mean()
    print(f"train loss: {avg_train_loss}")
    # -----------------------------

