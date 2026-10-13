import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import random

sys.path.append('src')
from models.fusion_model import FusionModelV2

def train_from_features():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print("Loading pre-extracted features...")
    data = torch.load('checkpoints/extracted_features.pt')
    features_tensor = data['features']
    labels_tensor = data['labels']
    
    print(f"Total Combined Images: {len(features_tensor)}")
    
    # Shuffle the features and labels randomly to mix the generic and face datasets
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(features_tensor))
    features_tensor = features_tensor[shuffled_indices]
    labels_tensor = labels_tensor[shuffled_indices]
    
    train_size = int(0.9 * len(features_tensor))
    train_features = features_tensor[:train_size]
    train_labels = labels_tensor[:train_size]
    val_features = features_tensor[train_size:]
    val_labels = labels_tensor[train_size:]
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 15
    print("Training robust classification head (with Frequency Dropout)...")
    for epoch in range(epochs):
        model.fusion_head.train()
        total_loss, correct, total = 0, 0, 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            # Apply Frequency Dropout! 50% chance to zero out the frequency features (last 128 dims)
            if random.random() < 0.5:
                # CLIP spatial is first 768 dims. Frequency is next 128.
                feats[:, 768:] = 0.0

            optimizer.zero_grad()
            outputs = model.fusion_head(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        model.fusion_head.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                
                # During validation, we also evaluate robustness by randomly dropping frequency half the time
                if random.random() < 0.5:
                    feats[:, 768:] = 0.0

                outputs = model.fusion_head(feats)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {correct/total*100:.2f}% | Val Acc: {val_correct/val_total*100:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_h_clip_robust.pt')
    print("Saved dual-robust model to checkpoints/model_h_clip_robust.pt")

if __name__ == "__main__":
    train_from_features()
