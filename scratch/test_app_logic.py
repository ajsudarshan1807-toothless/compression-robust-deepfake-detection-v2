import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

sys.path.append('src')
from models.fusion_model import FusionModelV2

def test_app_logic():
    device = torch.device("cpu")
    print("Loading image model...")
    image_model = FusionModelV2(pretrained_spatial=False, freeze_clip=True).to(device)
    img_model_path = "checkpoints/model_h_clip_robust.pt"
    if os.path.exists(img_model_path):
        image_model.load_state_dict(torch.load(img_model_path, map_location=device))
    else:
        print("Model not found!")
        return
    image_model.eval()
    
    mtcnn = MTCNN(keep_all=False, device=device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        test_dir = "deepfake-detector/test_images"
        
    images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        display_image = Image.open(img_path).convert('RGB')
        
        input_tensor_global = transform(display_image).unsqueeze(0).to(device)
        with torch.no_grad():
            score_global = image_model(input_tensor_global).item()
            
        boxes, _ = mtcnn.detect(display_image)
        score_face = None
        
        if boxes is not None:
            box = boxes[0]
            box = [int(b) for b in box]
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(display_image.width, box[2])
            box[3] = min(display_image.height, box[3])
            
            face_img = display_image.crop(box)
            input_tensor_face = transform(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                score_face = image_model(input_tensor_face).item()
                
        if score_face is not None:
            confidence_fake = max(score_global, score_face)
            print(f"{img_name}: Global={score_global:.4f}, Face={score_face:.4f} -> Fake Conf={confidence_fake:.4f}")
        else:
            confidence_fake = score_global
            print(f"{img_name}: Global={score_global:.4f}, Face=None -> Fake Conf={confidence_fake:.4f}")

if __name__ == "__main__":
    test_app_logic()
