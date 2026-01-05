import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from models.siamese_net import SiameseUNet
import os

# 1. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SiameseUNet().to(device)
model_path = 'models/best_model.pth'

if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Checking for latest_model.pth...")
    model_path = 'models/latest_model.pth'
    
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: No model checkpoint found. Inference will use random weights.")

model.eval()

# 2. Define Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_change(img_A, img_B):
    """
    Inference function for Gradio.
    Args:
        img_A, img_B: PIL Images (or numpy arrays converted by Gradio)
    Returns:
        mask: Binary change mask (numpy uint8)
        overlay: Overlay visualization (numpy uint8)
    """
    if img_A is None or img_B is None:
        return None, None

    # Convert to PIL if needed
    if not isinstance(img_A, Image.Image):
        img_A = Image.fromarray(img_A).convert('RGB')
    if not isinstance(img_B, Image.Image):
        img_B = Image.fromarray(img_B).convert('RGB')

    # Preprocess
    input_A = transform(img_A).unsqueeze(0).to(device)
    input_B = transform(img_B).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_A, input_B)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
    # Postprocess
    mask = (pred_prob > 0.5).astype(np.uint8) * 255
    
    # Create Overlay (Red on T2)
    # Resize T2 to 256x256 to match mask
    img_B_resized = img_B.resize((256, 256))
    img_B_np = np.array(img_B_resized)
    
    # Create red mask
    overlay = img_B_np.copy()
    red_mask = np.zeros_like(img_B_np)
    red_mask[:, :, 0] = 255 # Red channel
    
    # Blend: Where mask is white, blend with red
    alpha = 0.5
    mask_bool = mask > 0
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * red_mask[mask_bool]
    
    return mask, overlay

# 3. Create Gradio Interface
with gr.Blocks(title="Remote Sensing Change Detection") as demo:
    gr.Markdown("# 🌍 Remote Sensing Change Detection Demo")
    gr.Markdown("Upload two images (Time 1 and Time 2) to detect changes (e.g., new buildings).")
    
    with gr.Row():
        with gr.Column():
            img_input1 = gr.Image(label="Time 1 Image", type="pil")
            img_input2 = gr.Image(label="Time 2 Image", type="pil")
            btn_submit = gr.Button("Detect Changes", variant="primary")
            
        with gr.Column():
            img_output_mask = gr.Image(label="Change Mask", type="numpy")
            img_output_overlay = gr.Image(label="Visualization Overlay", type="numpy")
            
    btn_submit.click(fn=predict_change, inputs=[img_input1, img_input2], outputs=[img_output_mask, img_output_overlay])

    gr.Examples(
        examples=[], # Add example paths if available, e.g. [['data/test/A/1.png', 'data/test/B/1.png']]
        inputs=[img_input1, img_input2]
    )

if __name__ == "__main__":
    demo.launch()
