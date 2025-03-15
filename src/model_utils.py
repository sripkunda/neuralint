import os
import numpy as np
import torch
from data_utils import get_checkpoint_path

def load_model(model, checkpoint_dir="checkpoints", checkpoint_name="best_model.pth", optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try: 
        checkpoint = torch.load(get_checkpoint_path(checkpoint_dir, checkpoint_name), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] - 1
        loss = checkpoint.get('val_loss', checkpoint.get('train_loss', None))
        return epoch, loss
    except:
        return 0, float("inf")
    
def train(model, optimizer, train_loader, val_loader=None, num_epochs=100, checkpoint_dir="checkpoints", save_every = 10, starting_epoch = 0, best_val_loss = float("inf")):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Starting training at epoch {starting_epoch+1}/{num_epochs}, Validation Loss: {best_val_loss:.7f}")

    model.train()
    for epoch in range(starting_epoch, num_epochs):
        total_train_loss = 0
        for t, x, masks in train_loader:
            t, x, masks = t.to(device), x.to(device), masks.to(device)
            optimizer.zero_grad()
            transformer_out, neuralint_out, integral_function = model(t, x)
            loss = compute_loss(transformer_out, neuralint_out, x, integral_function, masks)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.7f}")
        
        if (epoch % save_every == 0):
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
            
            for t, x, masks in val_loader:
                t, x, masks = t.to(device), x.to(device), masks.to(device)
                optimizer.zero_grad()
                transformer_out, neuralint_out, integral_function = model(t, x)
                loss = compute_loss(transformer_out, neuralint_out, x, integral_function, masks)
                total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.7f}")
            
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

def compute_loss(transformer_out, neuralint_out, x, integral_function, masks):
    mse_loss = torch.nn.MSELoss()
    transformer_target_penalty = mse_loss(transformer_out * masks, x * masks)
    neuralint_transformer_penalty = mse_loss(neuralint_out, transformer_out)
    return 3/4 * neuralint_transformer_penalty + 1/4 * transformer_target_penalty + torch.mean(integral_function.T[0]**2)