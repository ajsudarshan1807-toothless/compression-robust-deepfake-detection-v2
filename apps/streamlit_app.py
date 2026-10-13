import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os
import cv2
import tempfile
import numpy as np

sys.path.append('src')
from compression_aug import jpeg_compress
from models.fusion_model import FusionModelV2
from models.video_model import TemporalFusionModel

# Pro Max Liquid Glass iOS Design Settings
st.set_page_config(page_title="Deepfake Shield", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS Injection for Liquid Glass iOS UI
st.markdown("""
<style>
    /* Typography: Apple System Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
    }
    
    /* Dynamic Mesh Gradient Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.15), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.15), transparent 25%),
                    radial-gradient(circle at 50% 80%, rgba(56, 189, 248, 0.15), transparent 25%),
                    linear-gradient(135deg, #020617 0%, #0f172a 100%);
        color: #f8fafc;
        background-attachment: fixed;
    }

    /* Hide default header/footer */
    header, footer {visibility: hidden;}

    /* Liquid Glassmorphism Panels */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 40px;
        box-shadow: 
            0 20px 40px -15px rgba(0,0,0,0.5), 
            inset 0 1px 0 rgba(255,255,255,0.1);
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .glass-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 
            0 30px 60px -20px rgba(0,0,0,0.6), 
            inset 0 1px 0 rgba(255,255,255,0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Typography Overhaul */
    h1 {
        font-weight: 800;
        font-size: 4rem !important;
        background: -webkit-linear-gradient(135deg, #38bdf8, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem !important;
        padding-top: 1rem !important;
        letter-spacing: -0.04em;
    }
    .sub-hero {
        text-align: center;
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 300;
        margin-bottom: 3.5rem;
        letter-spacing: -0.01em;
    }
    
    /* Styled Uploader with Neumorphic Inner Shadow */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(15, 23, 42, 0.4) !important;
        border: 1.5px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px;
        box-shadow: inset 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(56, 189, 248, 0.8) !important;
        background-color: rgba(56, 189, 248, 0.05) !important;
    }
    
    /* Input Fields (Login) */
    .stTextInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2), inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Streamlit Primary Button */
    [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.4) !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(129, 140, 248, 0.6) !important;
    }
    
    /* Glowing iOS Prediction Banners */
    .pred-fake {
        border: 1px solid rgba(239, 68, 68, 0.3);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
        text-align: center;
        animation: pulse-red 2s infinite alternate;
    }
    .pred-real {
        border: 1px solid rgba(34, 197, 94, 0.3);
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.05) 100%);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.15), inset 0 1px 0 rgba(255,255,255,0.1);
        text-align: center;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.15), inset 0 1px 0 rgba(255,255,255,0.1); }
        100% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.3), inset 0 1px 0 rgba(255,255,255,0.1); }
    }
    
    .pred-title {
        margin:0 0 8px 0; 
        font-weight: 800; 
        letter-spacing: -0.02em;
    }
    .pred-sub {
        margin:0; 
        font-size:20px; 
        color: rgba(255,255,255,0.8);
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Application ---
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # Load Image Model (Fusion Model V2)
    image_model = FusionModelV2(pretrained_spatial=False, freeze_clip=True).to(device)
    img_model_path = "checkpoints/model_i_whatsapp_robust.pt"
    if os.path.exists(img_model_path):
        image_model.load_state_dict(torch.load(img_model_path, map_location=device))
    else:
        st.warning("Image model weights not found!")
    image_model.eval()
    
    # Load Video Model (TemporalFusionModel)
    video_model = TemporalFusionModel(pretrained_spatial=False, freeze_clip=False, lstm_hidden_size=256).to(device)
    vid_model_path = "checkpoints/video_temporal_model_whatsapp.pt"
    if not os.path.exists(vid_model_path):
        vid_model_path = "checkpoints/video_temporal_model.pt"
    if os.path.exists(vid_model_path):
        try:
            payload = torch.load(vid_model_path, map_location=device)
            st_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            video_model.load_state_dict(st_dict)
        except Exception:
            pass
    video_model.eval()
    
    # Load MTCNN for Face Detection
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device=device)
    
    return image_model, video_model, mtcnn, device

image_model, video_model, mtcnn, device = load_models()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

st.markdown("<h1>Deepfake Shield</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-hero'>Dual-Stream Verification: Background & Face Authentication</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Image Detector", "🎥 Video Detector"])

with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Upload Image Media")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="img_upload")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        original_image = Image.open(uploaded_file).convert('RGB')
        
        quality = st.slider("Simulate Network Compression (JPEG QF)", min_value=10, max_value=100, value=100, step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Source**")
            st.image(original_image, use_container_width=True)
            
        if quality < 100:
            display_image = jpeg_compress(original_image, quality)
        else:
            display_image = original_image
            
        with col2:
            st.markdown(f"**Compressed (QF: {quality})**")
            st.image(display_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Dual-Stream Neural Analysis")
        
        with st.spinner("Analyzing global context and facial micro-artifacts..."):
            with st.expander("⚙️ Advanced Analysis Settings", expanded=False):
                threshold = st.slider("Detection Sensitivity (Threshold)", min_value=0.1, max_value=0.99, value=0.5, step=0.01, key="img_thresh")
                
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
                diagnostics = f"Global (Background) Score: {score_global:.4f} <br/> Local (Face) Score: {score_face:.4f}"
                st.markdown("#### Detected Face:")
                st.image(face_img, width=150)
            else:
                confidence_fake = score_global
                diagnostics = f"Global (Background) Score: {score_global:.4f} <br/> Local (Face) Score: N/A (No face detected)"
                
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2); padding:15px; border-radius:10px; margin-bottom:20px; font-family:monospace; color:#94a3b8;'>
                <b>Dual-Stream Diagnostics:</b><br/>
                {diagnostics}
            </div>
            """, unsafe_allow_html=True)
                
            if confidence_fake > threshold:
                st.markdown(f"""
                <div class='pred-fake'>
                    <h2 class='pred-title' style='color:#ef4444;'>⚠️ AI-Generated (Synthetic)</h2>
                    <p class='pred-sub'>Confidence: {confidence_fake*100:.1f}%</p>
                    <p style='font-size:14px; margin-top:10px; color:#fca5a5;'>*Splicing detected: the model found significant synthetic artifacts in the image.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-real'>
                    <h2 class='pred-title' style='color:#22c55e;'>✅ Authentic Image (Real)</h2>
                    <p class='pred-sub'>Confidence: {(1.0 - confidence_fake)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.progress(confidence_fake)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Upload Video Media")
    uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"], key="vid_upload")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_video is not None:
        if not os.path.exists("checkpoints/video_temporal_model_whatsapp.pt") and not os.path.exists("checkpoints/video_temporal_model.pt"):
            st.warning("⚠️ Temporal Video Model weights not found! The analysis will run with uninitialized (random) weights for demonstration. Run `train_video.py` to train the model on a video dataset.")
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.video(uploaded_video)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Temporal Sequence Analysis")
        
        with st.spinner("Extracting frames and analyzing temporal artifacts..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                vid_path = tfile.name
                
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            seq_len = 10
            stride = max(1, total_frames // seq_len)
            
            frames = []
            display_frames = []
            frame_idx = 0
            
            while cap.isOpened() and len(frames) < seq_len:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % stride == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    display_frames.append(img)
                    frames.append(transform(img))
                frame_idx += 1
            cap.release()
            os.remove(vid_path)
            
            while len(frames) < seq_len:
                pad_img = Image.new('RGB', (224, 224), (0, 0, 0))
                frames.append(transform(pad_img))
                display_frames.append(pad_img)
                
            sequence_tensor = torch.stack(frames).unsqueeze(0).to(device)
            
            st.markdown("#### Sampled Frames Sequence")
            st.image(display_frames, width=64)
            
            with torch.no_grad():
                video_score = video_model(sequence_tensor).item()
                
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2); padding:15px; border-radius:10px; margin-bottom:20px; font-family:monospace; color:#94a3b8;'>
                <b>Temporal Diagnostics:</b><br/>
                LSTM Sequence Score: {video_score:.4f}
            </div>
            """, unsafe_allow_html=True)
            
            threshold = 0.5
            if video_score > threshold:
                st.markdown(f"""
                <div class='pred-fake'>
                    <h2 class='pred-title' style='color:#ef4444;'>⚠️ AI-Generated Video (Deepfake)</h2>
                    <p class='pred-sub'>Confidence: {video_score*100:.1f}%</p>
                    <p style='font-size:14px; margin-top:10px; color:#fca5a5;'>*Temporal inconsistencies detected across frames.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-real'>
                    <h2 class='pred-title' style='color:#22c55e;'>✅ Authentic Video (Real)</h2>
                    <p class='pred-sub'>Confidence: {(1.0 - video_score)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.progress(video_score)
        st.markdown("</div>", unsafe_allow_html=True)
