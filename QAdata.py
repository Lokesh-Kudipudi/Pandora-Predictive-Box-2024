import torch
import torch.nn as nn
from torch.utils.data import  Dataset

def casual_mask(size):
  # Creating a square matrix of dimensions 'size x size' filled with ones
  mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
  return mask == 0

class MedicalDatasetQA(Dataset):

  def __init__(self, ds, tokenizer, seq_len) -> None:
    super().__init__()
    self.ds = ds
    self.tokenizer = tokenizer
    self.seq_len = seq_len

    self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
  
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, index):
    src_target_pair = self.ds[index]
    src_text = src_target_pair['question']
    tgt_text = src_target_pair['answer']

    # Tokenizing source and target texts 
    enc_input_tokens = self.tokenizer.encode(src_text).ids
    dec_input_tokens = self.tokenizer.encode(tgt_text).ids

    # Computing how many padding tokens need to be added to the tokenized texts
    # Source tokens
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # Subtracting the two '[EOS]' and '[SOS]' special tokens
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Subtracting the '[SOS]' special token

    # If the texts exceed the 'seq_len' allowed, it will raise an error. This means that one of the sentences in the pair is too long to be processed
    # given the current sequence length limit (this will be defined in the config dictionary below)
    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError('Sentence is too long')
    
    # Building the encoder input tensor by combining several elements
    # inserting the '[SOS]' token
    # Inserting the tokenized source text
    # Inserting the '[EOS]' token
    # Addind padding tokens
    encoder_input = torch.cat([self.sos_token,  torch.tensor(enc_input_tokens, dtype = torch.int64),self.eos_token,torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)])

    # Building the decoder input tensor by combining several elements
    # inserting the '[SOS]' token 
    # Inserting the tokenized target text
    # Addind padding tokens
    decoder_input = torch.cat([self.sos_token, torch.tensor(dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)])

    # Creating a label tensor, the expected output for training the model
    # Inserting the tokenized target text
    # Inserting the '[EOS]' token
    # Adding padding tokens
    label = torch.cat([torch.tensor(dec_input_tokens, dtype = torch.int64), self.eos_token, torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)])

    # Ensuring that the length of each tensor above is equal to the defined 'seq_len'
    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
      'encoder_input': encoder_input,
      'decoder_input': decoder_input, 
      'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
      'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), 
      'label': label,
      'src_text': src_text,
      'tgt_text': tgt_text
      }