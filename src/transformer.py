import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_attention_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_attention_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation="gelu")
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x):
        embedded_t = self.embedding(t)
        embedded_x = self.embedding(x)
        transformer_output = self.transformer(embedded_t, embedded_x)
        transformer_output = self.linear(transformer_output)
        return transformer_output