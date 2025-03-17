import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_attention_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Linear layer for embedding the input
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Transformer model
        self.transformer = nn.Transformer(hidden_dim, num_attention_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation="gelu")
        # Linear layer for projecting the output back to the input dimension
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x):
        # Embed the time points
        embedded_t = self.embedding(t)
        # Embed the input data
        embedded_x = self.embedding(x)
        # Pass the embedded inputs through the Transformer model
        transformer_output = self.transformer(embedded_t, embedded_x)
        # Project the Transformer output back to the input dimension
        transformer_output = self.linear(transformer_output)
        return transformer_output