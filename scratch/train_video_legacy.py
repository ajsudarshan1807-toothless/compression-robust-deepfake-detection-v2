import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append('src')
from models.video_model import TemporalFusionModel

class VideoDataset(Dataset):
    """
    Loads video files, extracts a sequence of frames, and prepares them.
    This is designed to load from a directory structure:
    data/videos/
        real/
            video1.mp4
        fake/
            video2.mp4
    """
    def __init__(self, root_dir, seq_len=10, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.video_paths = []
        self.labels = []
        
        # Load Real Videos (Label = 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(real_dir, file))
                    self.labels.append(0)
                    
        # Load Fake Videos (Label = 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(fake_dir, file))
                    self.labels.append(1)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate stride to get exactly `seq_len` frames evenly spaced
        stride = max(1, total_frames // self.seq_len)
        
        frames = []
        frame_idx = 0
        while cap.isOpened() and len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % stride == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
                
            frame_idx += 1
            
        cap.release()
        
        # Pad with black frames if video is too short
        while len(frames) < self.seq_len:
            pad_img = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                pad_img = self.transform(pad_img)
            frames.append(pad_img)
            
        # Stack into (seq_len, channels, height, width)
        sequence_tensor = torch.stack(frames)
        return sequence_tensor, torch.tensor([label], dtype=torch.float32)

def train_video_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Example Directory (Update with actual paths)
    dataset_path = 'data/videos'
    
    if not os.path.exists(dataset_path):
        print(f"Please create '{dataset_path}' and add 'real' and 'fake' folders with .mp4 files.")
        return
        
    dataset = VideoDataset(dataset_path, seq_len=10, transform=transform)
    
    if len(dataset) == 0:
        print("No videos found in dataset_path.")
        return
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Initializing TemporalFusionModel (LSTM)...")
    model = TemporalFusionModel(pretrained_spatial=True, freeze_clip=True, lstm_hidden_size=256).to(device)
    
    # Only train the LSTM and Video Head
    optimizer = optim.Adam(list(model.lstm.parameters()) + list(model.video_head.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 10
    print("Starting Video Training Pipeline...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for sequences, labels in progress:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({"Loss": total_loss/total, "Acc": correct/total})
            
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/video_temporal_model.pt')
    print("Saved Temporal Video model to checkpoints/video_temporal_model.pt")

if __name__ == '__main__':
    train_video_model()
