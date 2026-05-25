"""
Evaluation script for AI for Computational Mechanics (Assignment 1).
Handles testing on unseen data, plotting loss curves, and comparing predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def calculate_relative_error(predictions, targets):
    """
    Calculates the relative error as defined in the assignment:
    |E_predicted - E_true| / E_true
    """
    # Add a small epsilon to avoid division by zero in edge cases
    epsilon = 1e-8
    error = torch.abs(predictions - targets) / (targets + epsilon)
    return torch.mean(error).item()

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    total_error_sum = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            epsilon = 1e-8
            batch_errors = torch.abs(outputs - labels) / (labels + epsilon)
            
            total_error_sum += torch.sum(batch_errors).item()
            total_samples += labels.size(0)
            
    avg_error = total_error_sum / total_samples
    print(f"Test Set Average Relative Error: {avg_error * 100:.2f}%")
    
    if avg_error < 0.30:
        print("[Success]: Model meets the < 30% error requirement!")
    else:
        print("[Failure]: Model does not meet the 30% error requirement yet.")
        
    return avg_error

def plot_loss_curves(train_losses, val_losses, save_path="../report/loss_curves.pdf"):
    """
    Plots training and validation loss curves and saves them for the report.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Training Loss', color='black', linewidth=1.5)
    plt.plot(val_losses, label='Validation Loss', color='gray', linestyle='--', linewidth=1.5)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Loss curves saved to {save_path}")

def plot_predictions(model, test_loader, num_samples=10, device='cpu', save_path="../report/predictions.pdf"):
    """
    Randomly selects up to 10 data points and plots Predicted vs True values.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    # Collect all predictions and targets from the test set
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
            
    # Randomly sample 10 points
    total_points = len(all_preds)
    indices = random.sample(range(total_points), min(num_samples, total_points))
    
    sampled_preds = [all_preds[i] for i in indices]
    sampled_targets = [all_targets[i] for i in indices]
    
    # Bar chart comparison
    x = np.arange(len(sampled_preds))
    width = 0.35
    
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, sampled_targets, width, label='True $E_{eff}$', color='lightgray', edgecolor='black')
    plt.bar(x + width/2, sampled_preds, width, label='Predicted $E_{eff}$', color='dimgray', edgecolor='black')
    
    plt.ylabel("Young's Modulus (GPa)")
    plt.title('True vs. Predicted Effective Stiffness (10 Random Samples)')
    plt.xticks(x, [f"Sample {i+1}" for i in range(len(sampled_preds))], rotation=45)
    plt.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Prediction comparison saved to {save_path}")
