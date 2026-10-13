import torch
import sys

sys.path.append('src')
from models.video_model import TemporalFusionModel

def test_model():
    print("Initializing TemporalFusionModel...")
    # Use freeze_clip=True and a small lstm_hidden_size for faster testing
    model = TemporalFusionModel(pretrained_spatial=False, freeze_clip=True, lstm_hidden_size=64)
    
    # Create dummy video batch: (batch_size, sequence_length, channels, height, width)
    # Let's say batch=2 videos, seq_len=5 frames per video, 3 channels, 224x224
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    
    print(f"Input shape: {dummy_input.shape}")
    
    print("Running forward pass...")
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    print("Success! TemporalFusionModel compiles and runs forward pass.")

if __name__ == '__main__':
    test_model()
