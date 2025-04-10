import torch
import torch.nn as nn


# 简单Transformer编码器
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


class SimpleTransformerQA(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, heads=4, ff_hidden=512, num_layers=2, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(max_len, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, ff_hidden) for _ in range(num_layers)]
        )
        self.qa_outputs = nn.Linear(embed_dim, 2)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        pos = torch.arange(0, T).unsqueeze(0).expand(B, T).to(input_ids.device)
        x = self.embedding(input_ids) + self.position(pos)
        x = self.transformer(x)
        logits = self.qa_outputs(x)  # [B, T, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)