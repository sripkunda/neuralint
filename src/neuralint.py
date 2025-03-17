import os
import torch
import torch.nn as nn
import math
from model_utils import load_model
from transformer import TransformerModel

class NeuralInt(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_attention_heads=2, num_int_attention_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(NeuralInt, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, num_attention_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.diffintg = DiffIntg(input_dim, num_attention_heads=num_int_attention_heads)

    def forward(self, t, x):
        transformer_output = self.transformer(t, x)
        neural_int_output, integral_fn, _ = self.diffintg(transformer_output.shape[0])
        return transformer_output, neural_int_output, integral_fn - integral_fn[:, 0].unsqueeze(1)

class DiffIntg(nn.Module):
    def __init__(self, n_tmpts, hidden_dim=128, num_attention_heads=2, dim_feedforward=512):
        super().__init__()
        self.n_tmpts = n_tmpts
        self.num_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_attention_heads

        self.embedding = nn.Linear(1, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1)
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch_size):
        # Create time values from 0 to 1, with shape (batch_size, n_tmpts, 1)
        t = torch.linspace(0, 1, self.n_tmpts, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        t = t.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # Shape: (batch_size, n_tmpts, 1)
        t = t.detach().requires_grad_()

        # Project embeddings
        embedded_t = self.embedding(t)  # Shape: (batch_size, n_tmpts, hidden_dim)

        # Compute attention
        q = self.q_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_output = torch.matmul(attn_weights, v)  

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.n_tmpts, self.hidden_dim)
        integral_fn = self.out_proj(attn_output)  

        # Run through feedforward network
        integral_fn = self.output(integral_fn).view(batch_size, self.n_tmpts)
 
        # Compute gradients with respect to t
        grad_outputs = torch.ones_like(integral_fn)
        grad = torch.autograd.grad(outputs=integral_fn, inputs=t,
                                   grad_outputs=grad_outputs,
                                   create_graph=True, retain_graph=True)[0].view(batch_size, self.n_tmpts)

        return grad, integral_fn, t



def get_imputation_from_checkpoint(T, X, checkpoint_dir="model_checkpoints"):
    n_tpts = T.shape[1]
    model = NeuralInt(input_dim=n_tpts)
    epoch, loss = load_model(model, checkpoint_dir=checkpoint_dir)

    print(f"Loaded model checkpoint with epoch: {epoch}, and validation loss: {loss}")

    transformer_out, neuralint_out, integral_fn = model(T, X)
    return transformer_out, neuralint_out, integral_fn
