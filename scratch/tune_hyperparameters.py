import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import random

sys.path.append('src')
from models.fusion_model import FusionModelV2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for training: {device}")
    
    # 1. Load pre-extracted training features
    features_path = 'checkpoints/extracted_features.pt'
    if not os.path.exists(features_path):
        print(f"Features file {features_path} not found! Please run extract_features.py first.")
        return
        
    data = torch.load(features_path, map_location='cpu')
    features_tensor = data['features']
    labels_tensor = data['labels']
    print(f"Loaded training features: {features_tensor.shape}, labels: {labels_tensor.shape}")
    
    # Shuffle and split
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
    
    # 2. Extract and cache features for the test images
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        test_dir = "deepfake-detector/test_images"
        
    images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    mtcnn = MTCNN(keep_all=False, device=device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    fe_model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    fe_model.eval()
    
    test_cached_features = {}
    
    print("Caching test image features...")
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Global
        input_tensor_global = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            s_glob = fe_model.spatial(input_tensor_global, return_features=True)
            f_glob = fe_model.frequency(input_tensor_global, return_features=True)
            feat_glob = torch.cat((s_glob, f_glob), dim=1).cpu()
            
        # Face
        boxes, _ = mtcnn.detect(img)
        feat_face = None
        if boxes is not None:
            box = boxes[0]
            box = [int(b) for b in box]
            box[0], box[1] = max(0, box[0]), max(0, box[1])
            box[2], box[3] = min(img.width, box[2]), min(img.height, box[3])
            
            face_img = img.crop(box)
            input_tensor_face = transform(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                s_face = fe_model.spatial(input_tensor_face, return_features=True)
                f_face = fe_model.frequency(input_tensor_face, return_features=True)
                feat_face = torch.cat((s_face, f_face), dim=1).cpu()
                
        test_cached_features[img_name] = {
            'global': feat_glob,
            'face': feat_face
        }
    
    print("Test features cached successfully.\n")
    
    # 3. Define trial runner
    # We want to test different architectures and optimizer settings
    trials = [
        # Trial 1: Baseline (original architecture)
        {
            "name": "Baseline (Weight Decay 1e-4)",
            "hidden_dim": 512,
            "dropout": 0.5,
            "spatial_dropout": 0.0,
            "weight_decay": 1e-4,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 2: High Weight Decay (1e-2)
        {
            "name": "Weight Decay 1e-2",
            "hidden_dim": 512,
            "dropout": 0.5,
            "spatial_dropout": 0.0,
            "weight_decay": 1e-2,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 3: Very High Weight Decay (1e-1)
        {
            "name": "Weight Decay 1e-1",
            "hidden_dim": 512,
            "dropout": 0.5,
            "spatial_dropout": 0.0,
            "weight_decay": 1e-1,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 4: Smaller Hidden Dim (128)
        {
            "name": "Smaller Hidden Dim (128)",
            "hidden_dim": 128,
            "dropout": 0.5,
            "spatial_dropout": 0.0,
            "weight_decay": 1e-4,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 5: Single Linear Layer
        {
            "name": "Single Layer (Linear classifier)",
            "hidden_dim": None,  # Special case: direct linear projection
            "dropout": 0.0,
            "spatial_dropout": 0.0,
            "weight_decay": 1e-3,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 6: Spatial Feature Dropout (0.3)
        {
            "name": "Spatial Feature Dropout (0.3)",
            "hidden_dim": 512,
            "dropout": 0.5,
            "spatial_dropout": 0.3,
            "weight_decay": 1e-4,
            "lr": 1e-3,
            "epochs": 15
        },
        # Trial 7: Spatial Feature Dropout (0.5) + High Weight Decay (1e-2)
        {
            "name": "Spatial Dropout (0.5) + WD 1e-2",
            "hidden_dim": 512,
            "dropout": 0.5,
            "spatial_dropout": 0.5,
            "weight_decay": 1e-2,
            "lr": 1e-3,
            "epochs": 15
        }
    ]
    
    for trial in trials:
        print(f"==================================================")
        print(f"Running Trial: {trial['name']}")
        print(f"==================================================")
        
        # Build head
        fusion_dim = 896
        if trial['hidden_dim'] is None:
            head = nn.Sequential(
                nn.Linear(fusion_dim, 1),
                nn.Sigmoid()
            )
        else:
            head = nn.Sequential(
                nn.Linear(fusion_dim, trial['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(trial['dropout']),
                nn.Linear(trial['hidden_dim'], 1),
                nn.Sigmoid()
            )
            
        head = head.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=trial['lr'], weight_decay=trial['weight_decay'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Training loop
        for epoch in range(trial['epochs']):
            head.train()
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                
                # Frequency Dropout (from original pipeline)
                if random.random() < 0.5:
                    feats[:, 768:] = 0.0
                    
                # Spatial Feature Dropout (Trial config)
                if trial['spatial_dropout'] > 0.0:
                    # Apply dropout masking on spatial features
                    mask = (torch.rand(feats.size(0), 768, device=device) > trial['spatial_dropout']).float()
                    # Scale to maintain expected value
                    feats[:, :768] = feats[:, :768] * mask / (1.0 - trial['spatial_dropout'])
                
                optimizer.zero_grad()
                outputs = head(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        # Validation accuracy
        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = head(feats)
                pred_labels = (outputs > 0.5).float()
                val_correct += (pred_labels == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Evaluate on key test images
        key_images = [
            ("media__1785125869017.jpg", "Stalin Cutout Selfie (Real)"),
            ("media__1785125879078.jpg", "Two Friends Selfie (Real)"),
            ("media__1785133476155.jpg", "Gym Mirror Selfie (Real)"),
            ("media__1785125891983.png", "UI Screenshot (Fake)")
        ]
        
        print("\nPredictions on key test images:")
        for img_name, desc in key_images:
            feats_info = test_cached_features.get(img_name)
            if feats_info is None:
                continue
                
            f_glob = feats_info['global'].to(device)
            with torch.no_grad():
                score_glob = head(f_glob).item()
                
            score_face = 0.0
            if feats_info['face'] is not None:
                f_face = feats_info['face'].to(device)
                with torch.no_grad():
                    score_face = head(f_face).item()
                    
            max_score = max(score_glob, score_face)
            pred_class = "AI-Generated" if max_score > 0.5 else "Authentic"
            print(f"  - {img_name} ({desc}): Max Score: {max_score:.4f} -> Pred: {pred_class} (Global: {score_glob:.4f}, Face: {score_face:.4f})")
        print()

if __name__ == '__main__':
    main()
