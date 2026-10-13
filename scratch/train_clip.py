import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import os
import sys

sys.path.append('src')
from models.fusion_model import FusionModelV2

def train_clip():
    device = torch.device("cpu") # User is likely on CPU for training
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    # Load a smaller subset of the dataset just for rapid fine-tuning of the linear head
    # MUST shuffle before selecting, otherwise we only get Class 0!
    dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train')
    dataset = dataset.shuffle(seed=42).select(range(2000))
    dataset = dataset.train_test_split(test_size=0.1)
    
    # CLIP uses standard ImageNet normalization and 224x224 input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            # Dataset labels: 0 = AiArtData (Fake), 1 = RealArt (Real)
            # Our convention: 0 = Real, 1 = Fake
            label = 1 if item['label'] == 0 else 0
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    train_data = HFDataset(dataset['train'], transform)
    val_data = HFDataset(dataset['test'], transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print("Initializing FusionModelV2 (CLIP + Frequency)...")
    # freeze_clip=True ensures we don't destroy the robust ViT weights, we just train the classifier
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    
    # Only optimize the fusion head since CLIP is frozen
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    epochs = 2 # Since we are only training a single Linear layer, it converges instantly
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in progress:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({"Loss": total_loss/total, "Acc": correct/total})
            
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Val Acc: {val_correct/val_total*100:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_e_clip_fusion.pt')
    print("Saved CLIP Fusion model to checkpoints/model_e_clip_fusion.pt")

if __name__ == "__main__":
    train_clip()
