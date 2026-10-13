import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import sys
import io
sys.path.append('src')
from models.fusion_model import FusionModel
from compression_aug import jpeg_compress

df = pd.read_csv('data/splits/val.csv')
model = FusionModel(pretrained_spatial=False)
model.load_state_dict(torch.load('checkpoints/model_d_fusion_compaug.pt', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate(use_compress=False):
    real_preds = []
    fake_preds = []
    count = 0
    for _, row in df.iterrows():
        if count >= 100: break
        img = Image.open(row['filename']).convert('RGB')
        if use_compress:
            img = jpeg_compress(img, 90)
        
        out = model(transform(img).unsqueeze(0)).item()
        if row['label'] == 0:
            real_preds.append(out)
        else:
            fake_preds.append(out)
        count += 1
    
    print(f"--- Compress: {use_compress} ---")
    print('Real predictions:', sum(1 for x in real_preds if x < 0.5), '/', len(real_preds), '| Avg conf:', sum(real_preds)/len(real_preds) if real_preds else 0)
    print('Fake predictions:', sum(1 for x in fake_preds if x >= 0.5), '/', len(fake_preds), '| Avg conf:', sum(fake_preds)/len(fake_preds) if fake_preds else 0)

evaluate(use_compress=False)
evaluate(use_compress=True)
