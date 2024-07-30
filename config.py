from pathlib import Path
# Define settings for building and training the transformer model
def get_config():
  state = 'preTrain'   
  dataPath = ''
  batchSize = ''
  if state == 'preTrain':
    dataPath = 'context.json'
    batchSize = 32
  if state == 'fineTune.json':
    batchSize = 5
    dataPath = 'data.json'  

  return{
    'batch_size': batchSize,
    'num_epochs': 10,
    'lr': 10**-4,
    'seq_len': 350,
    'd_model': 512, # Dimensions of the embeddings in the Transformer. 512 like in the "Attention Is All You Need" paper.
    'model_folder': 'weights',
    'model_basename': 'tmodel_',
    'preload': None,
    'tokenizer_file': 'tokenizer.json',
    'experiment_name': 'runs/tmodel',
    'N': 8,
    'h': 8,
    'data_path': f"{dataPath}",
    'state': f"{state}",
    'dropout': 0.1,
    'd_ff': 2048
  }
    

# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
  model_folder = config['model_folder'] # Extracting model folder from the config
  model_basename = config['model_basename'] # Extracting the base name for model files
  model_filename = f"{model_basename}{epoch}.pt" # Building filename
  return str(Path('.')/ model_folder/ model_filename) # Combining current directory, the model folder, and the model filename
