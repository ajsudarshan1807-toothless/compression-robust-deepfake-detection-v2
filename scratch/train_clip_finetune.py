import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
import random
from PIL import Image
import io
import glob

sys.path.append('src')
from models.fusion_model import FusionModelV2

class HardFakesDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        # Use the images uploaded by the user in the brain directory
        brain_dir = r"C:\Users\sudarshan\.gemini\antigravity-ide\brain\cf955e02-7d14-4b84-aa7e-1ddd56ed693b"
        self.image_paths = glob.glob(os.path.join(brain_dir, "media__*.jpg"))
        
    def __len__(self):
        # We artificially multiply the size so the network sees them frequently
        return len(self.image_paths) * 20
        
    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        img = Image.open(self.image_paths[real_idx]).convert('RGB')
        # All of these are fakes, label = 1
        return self.transform(img), torch.tensor([1], dtype=torch.float32)

class HFDataset(Dataset):
    def __init__(self, hf_ds, transform, label_mapping_func):
        self.hf_ds = hf_ds
        self.transform = transform
        self.label_mapping_func = label_mapping_func
        
    def __len__(self):
        return len(self.hf_ds)
        
    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        image = item['image'].convert('RGB')
        label = self.label_mapping_func(item['label'])
        return self.transform(image), torch.tensor([label], dtype=torch.float32)

def train_clip_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    print("Loading Generic Dataset...")
    ds_generic = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train').shuffle(seed=42)
    ds_generic = ds_generic.select(range(min(10000, len(ds_generic))))
    generic_data = HFDataset(ds_generic, transform, lambda x: 1 if x == 0 else 0)
    
    print("Loading Specialized Face Dataset...")
    ds_face = load_dataset("Thien0103/DeepFake_Extracted_Face_Images", split='train').shuffle(seed=42)
    ds_face = ds_face.select(range(min(10000, len(ds_face))))
    face_data = HFDataset(ds_face, transform, lambda x: 1 if x == 0 else 0)
    
    print("Loading Hard Fakes (User Uploaded)...")
    hard_fakes = HardFakesDataset(transform)
    
    combined_data = torch.utils.data.ConcatDataset([generic_data, face_data, hard_fakes])
    loader = DataLoader(combined_data, batch_size=16, shuffle=True) # Small batch for memory
    
    print(f"Total training images: {len(combined_data)}")
    
    # Initialize with freeze_clip=False to allow fine-tuning
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=False).to(device)
    
    # Load previously trained robust head weights
    model_path = "checkpoints/model_h_clip_robust.pt"
    if os.path.exists(model_path):
        # strict=False because clip layer unfreezing doesn't change state_dict structure, 
        # but just in case
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded {model_path} to start fine-tuning.")
    
    # Only optimize the unfrozen parameters (Last 2 layers of CLIP + Fusion Head)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=1e-5, weight_decay=1e-4) # very small learning rate for fine-tuning
    criterion = nn.BCELoss()
    
    epochs = 10
    print("Starting Massive Fine-Tuning (End-to-End)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
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
            
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_h_clip_robust.pt') # Overwrite with fine-tuned version
    print("Saved fine-tuned robust model to checkpoints/model_h_clip_robust.pt")

if __name__ == '__main__':
    train_clip_finetune()
