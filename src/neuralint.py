import torch
import torch.nn as nn
from neuralint_utils import load_model
from transformer import TransformerModel

class NeuralInt(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_attention_heads=2, num_int_attention_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(NeuralInt, self).__init__()
        # Initialize the Transformer model
        self.transformer = TransformerModel(input_dim, hidden_dim, num_attention_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        # Initialize the Diff(Intg(t)) module for NeuralInt
        self.diffintg = DiffIntg(input_dim, num_attention_heads=num_int_attention_heads)

    def forward(self, t, x):
        # Pass the input through the Transformer model
        transformer_output = self.transformer(t, x)
        # Compute the output from NeuralInt
        neural_int_output, integral_fn, _ = self.diffintg(transformer_output.shape[0])
        # Return the outputs and the integral function adjusted so that Intg(0) = 0.
        return transformer_output, neural_int_output, integral_fn - integral_fn[:, 0].unsqueeze(1)

class DiffIntg(nn.Module):
    def __init__(self, n_tmpts, hidden_dim=128, num_attention_heads=2, dim_feedforward=512):
        super().__init__()
        self.n_tmpts = n_tmpts  # Number of time points
        self.num_heads = num_attention_heads  # Number of attention heads
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.head_dim = hidden_dim // num_attention_heads  # Dimension per attention head

        # Linear layer for embedding time points
        self.embedding = nn.Linear(1, hidden_dim)
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1)
        )
        # Linear layers for query, key, and value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch_size):
        # Generate time points for each batch
        t = torch.linspace(0, 1, self.n_tmpts, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        t = t.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        t = t.detach().requires_grad_()
        # Affine map to map t to the hidden dimension of the attention mechanism
        embedded_t = self.embedding(t) 
        # Project the embedded time points to queries, keys, and values
        q = self.q_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(embedded_t).view(batch_size, self.n_tmpts, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.n_tmpts, self.hidden_dim)
        # Compute the attention output and apply the FNN
        integral_fn = self.out_proj(attn_output)  
        integral_fn = self.output(integral_fn).view(batch_size, self.n_tmpts)
        # Compute gradients with respect to t
        grad_outputs = torch.ones_like(integral_fn)
        grad = torch.autograd.grad(outputs=integral_fn, inputs=t,
                                   grad_outputs=grad_outputs,
                                   create_graph=True, retain_graph=True)[0].view(batch_size, self.n_tmpts)

        return grad, integral_fn, t

def get_imputation_from_checkpoint(T, X, checkpoint_dir="model_checkpoints"):
    # Get the number of time points
    n_tpts = T.shape[1]
    # Initialize the model
    model = NeuralInt(input_dim=n_tpts)
    # Load the model from the checkpoint
    epoch, loss = load_model(model, checkpoint_dir=checkpoint_dir)
    print(f"Loaded model checkpoint with epoch: {epoch}, and validation loss: {loss}")
    # Pass the input through the model
    transformer_out, neuralint_out, integral_fn = model(T, X)
    return transformer_out, neuralint_out, integral_fn
