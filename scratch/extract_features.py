import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

sys.path.append('src')

from models.fusion_model import FusionModelV2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    class GenericDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            label = 1 if item['label'] == 0 else 0 
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    class FaceDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            label = 1 if item['label'] == 0 else 0 
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    print("Loading Generic Dataset (10,000 images)...")
    ds_generic = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train')
    ds_generic = ds_generic.shuffle(seed=42).select(range(10000))
    generic_data = GenericDataset(ds_generic, transform)
    
    print("Loading Specialized Face Dataset (Thien0103)...")
    ds_face = load_dataset("Thien0103/DeepFake_Extracted_Face_Images", split='train')
    ds_face = ds_face.shuffle(seed=42)
    face_data = FaceDataset(ds_face, transform)
    
    combined_data = torch.utils.data.ConcatDataset([generic_data, face_data])
    loader = DataLoader(combined_data, batch_size=64, shuffle=False)
    
    print(f"Total Combined Images: {len(combined_data)}")
    
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features into RAM...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            spatial_features = model.spatial(imgs, return_features=True)
            frequency_features = model.frequency(imgs, return_features=True)
            fused = torch.cat((spatial_features, frequency_features), dim=1)
            
            all_features.append(fused.cpu())
            all_labels.append(labels)
            
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/extracted_features.pt'
    torch.save({'features': features_tensor, 'labels': labels_tensor}, save_path)
    print(f"Saved pre-extracted features to {save_path}")

if __name__ == '__main__':
    main()
