import torch
import torch.nn as nn
import math
import numpy as np

class NonNegativeSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply non-negative sinusoidal encoding (NN-SPE) as described in the paper
        # Ensure values are in range [0,1] by scaling with 0.5 and adding 1
        pe[:, 0::2] = 0.5 * (torch.sin(position * div_term) + 1)
        pe[:, 1::2] = 0.5 * (torch.cos(position * div_term) + 1)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Adding positional encoding to word embeddings as described in the paper
        return x + self.pe[:, :x.size(1)]

class HybridAttentionMechanism(nn.Module):
    def __init__(self, hidden_dim, local_window_size=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local_window_size = local_window_size
        
        # Linear transformations for Q, K, V as described in the paper
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Gating mechanism for combining global and local attention
        self.gate = nn.Linear(hidden_dim, 1)
        
        # Scaling factor for dot-product attention
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Calculate Q, K, V projections
        Q = self.query(hidden_states)  # [batch_size, seq_len, hidden_dim]
        K = self.key(hidden_states)    # [batch_size, seq_len, hidden_dim]
        V = self.value(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate dot-product attention scores
        # ξᵢ = QᵢK^T/sqrt(d_k) as per paper equation (6)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(hidden_states.device)
        
        # Global attention - equation (7) from paper
        # Att(ξᵢ,V) = softmax(ξᵢ)·V
        global_attention = torch.softmax(attention_scores, dim=-1)
        global_output = torch.matmul(global_attention, V)
        
        # Local attention - equation (8) from paper - 修正掩碼實現
        local_attention = attention_scores.clone()
        
        # 完全重寫位置掩碼建立邏輯，更簡潔且正確處理維度
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        
        # 創建位置索引
        positions = torch.arange(seq_len, device=device)
        
        # 計算所有位置對之間的距離
        positions_i = positions.unsqueeze(1)  # [seq_len, 1]
        positions_j = positions.unsqueeze(0)  # [1, seq_len]
        relative_positions = positions_i - positions_j  # [seq_len, seq_len]
        
        # 創建窗口掩碼 (-local_window_size <= x <= local_window_size) -> True
        # 這會產生一個形狀為 [seq_len, seq_len] 的掩碼
        window_mask = (relative_positions.abs() <= self.local_window_size).float()
        
        # 擴展到批量維度 [batch_size, seq_len, seq_len]
        window_mask = window_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        # 將窗口外的注意力分數設為負無窮
        local_attention_mask = (1.0 - window_mask) * -1e9
        
        # 應用掩碼後再softmax
        local_attention = local_attention + local_attention_mask
        local_attention = torch.softmax(local_attention, dim=-1)
        local_output = torch.matmul(local_attention, V)
        
        # Gating mechanism - equation (9) from paper
        # gᵢ = σ(Whᵢ)
        gate = torch.sigmoid(self.gate(hidden_states))
        
        # Combine global and local attention - equation (10) from paper
        # Oᵢ = (1-gᵢ)×Att(ξᵢ,Vᵢ) + gᵢ×Att(B(ξᵢ),Vᵢ)
        output = (1 - gate) * global_output + gate * local_output
        
        return output

class PosAttBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, max_len=5000):
        super().__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Positional encoding layer - NN-SPE
        self.pos_encoding = NonNegativeSinusoidalPositionalEncoding(embedding_dim, max_len)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(embedding_dim, 
                             hidden_dim, 
                             num_layers=n_layers, 
                             bidirectional=bidirectional, 
                             batch_first=True, 
                             dropout=dropout if n_layers > 1 else 0)
        
        # Dimension for BiLSTM output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dimension reduction layer after BiLSTM (as mentioned in the paper)
        self.dim_reduction = nn.Linear(lstm_output_dim, hidden_dim)
        
        # Hybrid attention mechanism layer 
        self.hybrid_attention = HybridAttentionMechanism(hidden_dim, local_window_size=30)
        
        # 修改最終分類層以適應連接池化的輸出維度
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 現在是hidden_dim*2因為我們連接了兩個池化結果
        
        self.dropout = nn.Dropout(dropout)
        
        # 添加批量正規化
        self.bn = nn.BatchNorm1d(hidden_dim * 2)  # 在初始化時需更新
        
        # 應用權重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型權重以加速收斂"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM權重使用正交初始化
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:  # 確保參數至少是2維的
                    # 線性層使用xavier初始化，只對至少2維的參數使用
                    nn.init.xavier_uniform_(param)
                else:
                    # 對於1維權重，使用均勻分佈初始化
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, text):
        # Word embedding layer
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding - NN-SPE
        embedded = self.pos_encoding(embedded)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        # BiLSTM layer
        output, (hidden, cell) = self.bilstm(embedded)
        
        # Dimension reduction
        reduced_output = self.dim_reduction(output)
        
        # Apply hybrid attention
        attended = self.hybrid_attention(reduced_output)
        
        # 實現論文中的最大池化和平均池化結合
        # 1. 最大池化 - 捕捉最突出的特徵
        max_pooled = torch.max(attended, dim=1)[0]  # [batch_size, hidden_dim]
        
        # 2. 平均池化 - 捕捉整體特徵分佈
        mean_pooled = torch.mean(attended, dim=1)  # [batch_size, hidden_dim]
        
        # 3. 連接兩種池化結果
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)  # [batch_size, hidden_dim*2]
        
        # 應用批量正規化 - 需要調整 bn 層的維度
        pooled = self.bn(pooled)
        
        # 刪除或注釋掉原來的注意力加權池化代碼:
        # attention_weights = torch.softmax(torch.sum(attended, dim=2), dim=1).unsqueeze(2)
        # pooled = torch.sum(attended * attention_weights, dim=1)
        
        # Final classification
        return self.fc(pooled)