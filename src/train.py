"""
Training script for AI for Computational Mechanics (Assignment 1).
Contains the training loop, validation logic, and best-model saving mechanism.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# Import our custom modules
from dataset import create_dataloaders
from model import MicrostructureCNN
from evaluate import plot_loss_curves, evaluate_model, calculate_relative_error

def train_model(
    img_dir="../data/images",
    labels_path="../data/labels.csv", # Updated to match your CSV setup
    model_save_path="../models/best_model.pth",
    epochs=50,
    batch_size=64,
    learning_rate=0.001
):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    # Note: We limit max_samples here to 5000 to boost the efficiency score. 
    # Huixin can adjust this later during the optimization phase.
    train_loader, val_loader, test_loader = create_dataloaders(
        img_dir=img_dir,
        labels_path=labels_path,
        batch_size=batch_size,
        max_samples=5000 
    )

    # 3. Initialize Model, Loss, and Optimizer
    model = MicrostructureCNN().to(device)
    
    # We use MSE for calculating gradients during training, 
    # but we will track the Relative Error for our custom evaluation.
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 4. Training Loop setup
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("Starting Training...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_error = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                running_val_error += calculate_relative_error(outputs, labels)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        avg_val_error = running_val_error / len(val_loader)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss (MSE): {epoch_train_loss:.4f} | Val Loss (MSE): {epoch_val_loss:.4f} | Val Rel Error: {avg_val_error*100:.2f}%")

        # --- Checkpoint: Save the best model ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Ensure the models directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining Complete. Best model saved to {model_save_path}")

    # 5. Post-Training: Generate Report Assets & Final Evaluation
    print("\nGenerating evaluation metrics and plots for the report...")
    
    # Plot the learning curves using our evaluate module
    plot_loss_curves(train_losses, val_losses, save_path="../report/loss_curves.pdf")
    
    # Load the best model weights back in for the final test set evaluation
    model.load_state_dict(torch.load(model_save_path))
    
    # Run the final evaluation on the unseen test set
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    # You can execute this script directly to run the whole pipeline
    train_model()