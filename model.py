import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tensorboardX import SummaryWriter

import math


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from typing import Any
from tqdm import tqdm

import warnings

class InputEmbeddings(nn.Module):

  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model) # Converting Text to embeddings

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model) # Normalization 

class PositionalEncoding(nn.Module):

  #seq_len is the maximum length of the sentence
  def __init__(self, d_model:int, seq_len:int, dropout: float)->None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout) # Dropout Layer to prevent overfitting

    pe = torch.zeros(seq_len, d_model)

    #create a vector of (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    # Create a vector of shape (d_model)
    div_term = torch.exp(torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)))
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
    #Appy sine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
    #Adding the Batch Dimension
    pe = pe.unsqueeze(0) # (1, seq_len, d_model)
    #Register the positional encoding as a Buffer
    self.register_buffer('pe', pe)

  def forward(self,x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
    return self.dropout(x)
    
class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.ones(1))

  def forward(self, x):
    # x: (batch, seq_len, hidden_size)
    # Keep the dimension for broadcasting
    mean = x.mean(dim = -1, keepdim = True)
    std = x.std(dim = -1, keepdim = True)
    # eps is to prevent dividing by zero or when std is very small
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
  def __init__(self, d_model:int, d_ff:int, dropout:float)-> None:
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self,x):
    # x:(Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model:int, h:int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h # Number of heads
    assert d_model % h == 0 # For equal parts
    
    self.d_k = d_model // h # Dimension of vector seen by each head
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    # Just apply the formula from the paper
    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      # Write a very low value (indicating -inf) to the positions where mask == 0
      attention_scores.masked_fill_(mask==0, -1e9)
    # (batch, h, seq_len, seq_len) # Apply softmax  
    attention_scores = attention_scores.softmax(dim=-1)
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
    # return attention scores which can be used for visualization
    return (attention_scores@value), attention_scores


  def forward(self, q,k,v, mask):
    query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
    key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
    value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

    # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch,h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

    # Calculate attention
    x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

    # Combine all the heads together
    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x=x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

    # Multiply by Wo
    # (batch, seq_len, d_model) --> (batch, seq_len, d_model) 
    return self.w_o(x)

class ResidualConnection(nn.Module):

  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

  # This block takes in the MultiHeadAttentionBlock and FeedForwardBlock, as well as the dropout rate for the residual connections
  def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock, dropout: float) -> None:
    super().__init__()

    # Storing the self-attention block and feed-forward block
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    
    # 2 Residual Connections with dropout
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self,x, src_mask):
    # Applying the first residual connection with the self-attention block
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
    # Applying the second residual connection with the feed-forward block 
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x
  
class Encoder(nn.Module):

  # The Encoder takes in instances of 'EncoderBlock'
  def __init__(self, layer:nn.ModuleList)-> None:
    super().__init__()
    # Storing the EncoderBlocks
    self.layers = layer
    # Layer for the normalization of the output of the encoder layers
    self.norm = LayerNormalization()

  def forward(self, x, mask):
    # Iterating over each EncoderBlock stored in self.layers
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

# Building Decoder Block
class DecoderBlock(nn.Module):
  # The DecoderBlock takes in two MultiHeadAttentionBlock. One is self-attention, while the other is cross-attention.
  # It also takes in the feed-forward block and the dropout rate
  def __init__(self,  self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # List of three Residual Connections with dropout rate
      
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    # Self-Attention block with query, key, and value plus the target language mask
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    
    # The Cross-Attention block using two 'encoder_ouput's for key and value plus the source language mask. It also takes in 'x' for Decoder queries
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    
    # Feed-forward block with residual connections
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x
    
# Building Decoder
# A Decoder can have several Decoder Blocks
class Decoder(nn.Module):
  # The Decoder takes in instances of 'DecoderBlock'
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    # Storing the 'DecoderBlock's
    self.layers = layers
    self.norm = LayerNormalization() # Layer to normalize the output
      
  def forward(self, x, encoder_output, src_mask, tgt_mask):
    # Iterating over each DecoderBlock stored in self.layers
    for layer in self.layers:
      # Applies each DecoderBlock to the input 'x' plus the encoder output and source and target masks
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x) # Returns normalized output

class ProjectionLayer(nn.Module):
  def __init__(self, d_model, vocab_size)->None:
    super().__init__()
    # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    # Applying the log Softmax function to the output for probabilitites
    return torch.log_softmax(self.proj(x), dim=-1)

# Creating the Transformer Architecture
class Transformer(nn.Module):
  # This takes in the encoder and decoder, as well the embeddings for the source and target language.
  # It also takes in the Positional Encoding for the source and target language, as well as the projection layer
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projection_layer
      
  # Encoder     
  def encode(self, src, src_mask):
    src = self.src_embed(src) # Applying source embeddings to the input source language
    src = self.src_pos(src) # Applying source positional encoding to the source embeddings
    return self.encoder(src, src_mask) # Returning the source embeddings plus a source mask to prevent attention to certain elements
  
  # Decoder
  def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
    tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings
    
    # Returning the target embeddings, the output of the encoder, and both source and target masks
    # The target mask ensures that the model won't 'see' future elements of the sequence
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
  
  # Applying Projection Layer with the Softmax function to the Decoder output
  def project(self, x):
    return self.projection_layer(x)

# Building & Initializing Transformer
# Definin function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
  
  # Creating Embedding layers
  src_embed = InputEmbeddings(d_model, vocab_size) # Source language (Source Vocabulary to 512-dimensional vectors)
  tgt_embed = InputEmbeddings(d_model, vocab_size) # Target language (Target Vocabulary to 512-dimensional vectors)
  
  # Creating Positional Encoding layers
  src_pos = PositionalEncoding(d_model, seq_len, dropout) # Positional encoding for the source language embeddings
  tgt_pos = PositionalEncoding(d_model, seq_len, dropout) # Positional encoding for the target language embeddings
  
  # Creating EncoderBlocks
  encoder_blocks = [] # Initial list of empty EncoderBlocks
  for _ in range(N): # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
    encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout) # Self-Attention
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
    
    # Combine layers into an EncoderBlock
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block) # Appending EncoderBlock to the list of EncoderBlocks
      
  # Creating DecoderBlocks
  decoder_blocks = [] # Initial list of empty DecoderBlocks
  for _ in range(N): # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
    decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout) # Self-Attention
    decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout) # Cross-Attention
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
    
    # Combining layers into a DecoderBlock
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block) # Appending DecoderBlock to the list of DecoderBlocks
      
  # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))
  
  # Creating projection layer
  projection_layer = ProjectionLayer(d_model, vocab_size) # Map the output of Decoder to the Target Vocabulary Space
  
  # Creating the transformer by combining everything above
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
  
  # Initialize the parameters
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
          
  return transformer # Assembled and initialized Transformer. Ready to be trained and validated!