import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from config import get_config, get_weights_file_path
from tensorboardX import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
from validation import run_validation


from model import build_transformer
import warnings

import random
from dataset import MedicalDataset, casual_mask
import json

# Iterating through dataset to extract the original sentence and its translation 
def get_all_sentences(ds):
  for pair in ds:
    yield pair['question']
    yield pair['answer']

def get_or_build_tokenizer(config, ds):
   # Crating a file path for the tokenizer
  tokenizer_path = Path(config['tokenizer_file'])

  if not Path.exists(tokenizer_path):

    # If it doesn't exist, we create a new one
    tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]')) # Initializing a new world-level tokenizer
    tokenizer.pre_tokenizer = Whitespace() # We will split the text into tokens based on whitespace

    # Defining Word Level strategy and special tokens
    trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]","[SOS]", "[EOS]"], min_frequency=2)

    # Training new tokenizer on sentences from the dataset and language specified
    tokenizer.train_from_iterator(get_all_sentences(ds), trainer = trainer)
    # Saving trained tokenizer to the file path specified at the beginning of the function
    tokenizer.save(str(tokenizer_path)) 
  else:
    # If the tokenizer already exist, we load it
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer # Returns the loaded tokenizer or the trained tokenizer


def get_ds(config):

  # List of objects which contains the quesition and the asnwer 
  ds_raw = []
  with open(config['data_path'], 'r') as data:
    ds_raw = json.loads(data.read())

  #ds_raw = load_dataset("Helsinki-NLP/opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
  tokenizer = get_or_build_tokenizer(config, ds_raw)

  random.shuffle(ds_raw)

  # Splitting Data into training and Validation
  train_ds_size = int(0.9*len(ds_raw))
  
  #val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw = ds_raw[:train_ds_size]
  val_ds_raw = ds_raw[train_ds_size:]

  train_ds = MedicalDataset(train_ds_raw, tokenizer, config['seq_len'])
  val_ds = MedicalDataset(val_ds_raw, tokenizer, config['seq_len'])

  

  max_len_src = 0
  max_len_tgt = 0
  for pair in ds_raw:
    src_ids = tokenizer.encode(pair['question']).ids
    tgt_ids = tokenizer.encode(pair['answer']).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of Target Text sentence: {max_len_tgt}')

  # Creating dataloaders for the training and validadion sets
  # Dataloaders are used to iterate over the dataset in batches during training and validation
  # Batch size will be defined in the config dictionary
  train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
  val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
  return train_dataloader, val_dataloader, tokenizer # Returning the DataLoader objects and tokenizers

def get_model(config, vocab_len):
  model = build_transformer(vocab_len,config['seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
  return model

def train_model(config):
  #Define the Device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device {device}")

  # Creating model directory to store weights
  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
  # Retrieving dataloaders and tokenizers for source and target languages using the 'get_ds' function
  train_dataloader, val_dataloader, tokenizer = get_ds(config)

  # Initializing model on the GPU using the 'get_model' function
  model = get_model(config,tokenizer.get_vocab_size()).to(device)

  # Tensorboard
  writer = SummaryWriter(config['experiment_name'])

  # Setting up the Adam optimizer with the specified learning rate from the
  # config dictionary plus an epsilon value
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

  # Initializing epoch and global step variables
  initial_epoch = 0
  global_step = 0

  # Checking if there is a pre-trained model to load
  # If true, loads it
  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename) # Loading model

    # Sets epoch to the saved in the state plus one, to resume from where it stopped
    initial_epoch = state['epoch'] + 1
    # Loading the optimizer state from the saved model
    optimizer.load_state_dict(state['optimizer_state_dict'])
    # Loading the global step state from the saved model
    global_step = state['global_step']

  # Initializing CrossEntropyLoss function for training
  # We ignore padding tokens when computing loss, as they are not relevant for the learning process
  # We also apply label_smoothing to prevent overfitting
  loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

  # Initializing training loop 
  # Iterating over each epoch from the 'initial_epoch' variable up to
  # the number of epochs informed in the config
  for epoch in range(initial_epoch, config['num_epochs']):
    # Initializing an iterator over the training dataloader
    # We also use tqdm to display a progress bar
    batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')

    for batch in batch_iterator:
      model.train() # Train the model

      # Loading input data and masks onto the GPU
      encoder_input = batch['encoder_input'].to(device)
      decoder_input = batch['decoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)
      decoder_mask = batch['decoder_mask'].to(device)

      # Running tensors through the Transformer
      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
      proj_output = model.project(decoder_output)

      # Loading the target labels onto the GPU
      label = batch['label'].to(device)

      # Computing loss between model's output and true labels
      loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))

      # Updating progress bar
      batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

      writer.add_scalar('train loss', loss.item(), global_step)
      writer.flush()

      # Performing backpropagation
      loss.backward()

      # Updating parameters based on the gradients
      optimizer.step()

      # Clearing the gradients to prepare for the next batch
      optimizer.zero_grad()

      global_step += 1 # Updating global step count

      # We run the 'run_validation' function at the end of each epoch
      # to evaluate model performance

      run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

      model_filename = get_weights_file_path(config, f'{epoch:02d}')
      # Writting current model state to the 'model_filename'
      torch.save({
          'epoch': epoch, # Current epoch
          'model_state_dict': model.state_dict(),# Current model state
          'optimizer_state_dict': optimizer.state_dict(), # Current optimizer state
          'global_step': global_step # Current global step 
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore') # Filtering warnings
    config = get_config() # Retrieving config settings
    train_model(config) # Training model with the config arguments