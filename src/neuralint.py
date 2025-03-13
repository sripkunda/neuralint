import torch
import torch.nn as nn
import math
from data_utils import get_checkpoint_path

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, 1, hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NeuralInt(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512, num_attention_heads = 2, num_int_attention_heads = 10, num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = 2048, dropout=0.1):
        super(NeuralInt, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_attention_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation="gelu")
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.intg = Intg(input_dim, num_int_attention_heads)
        self.diff = Diff()


    def forward(self, t, x):
        embedded_t = self.embedding(t)
        embedded_x = self.embedding(x)
        output = self.transformer(embedded_t, embedded_x)
        output = self.linear(output)
        return output

class Intg(nn.Module):
    def __init__(self, input_dim, num_attention_heads = 10):
        super().__init__()

class Diff(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        input.grad = None
        input.backward(retain_graph=True)
        return input.grad

def train(model, loss_fn, train_loader, val_loader=None, num_epochs=10, learning_rate=3e-4, weight_decay=1e-8, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for t, x, masks in train_loader:
            t, x, masks = t.to(device), x.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(t, x)
            
            # Apply mask to compute loss only for observed values
            loss = loss_fn(outputs * masks, x * masks)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint after every epoch
        checkpoint_path = get_checkpoint_path(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Validation Step
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for t, x, masks in val_loader:
                    t, x, masks = t.to(device), x.to(device), masks.to(device)
                    outputs = model(t, x)
                    loss = loss_fn(outputs * masks, x * masks)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            
            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, best_model_path)
                print(f"Best model saved: {best_model_path}")
            
            model.train()