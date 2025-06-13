import torch
import torch.nn as nn
import math
import numpy as np

# Non-Negative Sinusoidal Positional Encoding Reproduction
class NonNegativeSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = 0.5 * (torch.sin(position * div_term) + 1)
        pe[:, 1::2] = 0.5 * (torch.cos(position * div_term) + 1)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Hybrid Attention Mechanism Reproduction
class HybridAttentionMechanism(nn.Module):
    def __init__(self, hidden_dim, local_window_size=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local_window_size = local_window_size
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.gate = nn.Linear(hidden_dim, 1)
        
        self.scale = math.sqrt(hidden_dim)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        global_attention = torch.softmax(attention_scores, dim=-1)
        global_output = torch.matmul(global_attention, V)
        
        local_attention = attention_scores.clone()
        
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        
        positions = torch.arange(seq_len, device=device)
        
        positions_i = positions.unsqueeze(1)
        positions_j = positions.unsqueeze(0)
        relative_positions = positions_i - positions_j
        
        window_mask = (relative_positions.abs() <= self.local_window_size).float()
        
        window_mask = window_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        local_attention_mask = (1.0 - window_mask) * -1e9
        
        local_attention = local_attention + local_attention_mask
        local_attention = torch.softmax(local_attention, dim=-1)
        local_output = torch.matmul(local_attention, V)
        
        gate = torch.sigmoid(self.gate(hidden_states))
        
        output = (1 - gate) * global_output + gate * local_output
        
        return output

class PosAttBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, max_len=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.pos_encoding = NonNegativeSinusoidalPositionalEncoding(embedding_dim, max_len)
        
        self.bilstm = nn.LSTM(embedding_dim, 
                             hidden_dim, 
                             num_layers=n_layers, 
                             bidirectional=bidirectional, 
                             batch_first=True, 
                             dropout=dropout if n_layers > 1 else 0)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dim_reduction = nn.Linear(lstm_output_dim, hidden_dim)
        
        self.hybrid_attention = HybridAttentionMechanism(hidden_dim, local_window_size=30)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for faster convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, text):
        # Word embedding
        embedded = self.embedding(text)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        embedded = self.dropout(embedded)
        
        # BiLSTM
        output, (hidden, cell) = self.bilstm(embedded)
        
        # Dimension reduction
        reduced_output = self.dim_reduction(output)
        
        # Apply hybrid attention
        attended = self.hybrid_attention(reduced_output)
        
        # Max and mean pooling combination
        max_pooled = torch.max(attended, dim=1)[0]
        mean_pooled = torch.mean(attended, dim=1)
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        
        # Batch normalization
        pooled = self.bn(pooled)
        
        return self.fc(pooled)
