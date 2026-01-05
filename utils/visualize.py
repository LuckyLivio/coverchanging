import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def visualize_result(img_A, img_B, label, pred_logits, save_path=None):
    """
    Visualize T1, T2, GT, Pred.
    
    Args:
        img_A, img_B: Tensor (C, H, W), normalized.
        label: Tensor (1, H, W).
        pred_logits: Tensor (1, H, W).
        save_path: Path to save the figure.
    """
    # Helper to denormalize
    def denorm(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor.cpu() * std + mean

    img_A = denorm(img_A).permute(1, 2, 0).numpy()
    img_B = denorm(img_B).permute(1, 2, 0).numpy()
    
    label = label.cpu().squeeze().numpy()
    pred_prob = torch.sigmoid(pred_logits).cpu().squeeze().numpy()
    pred_mask = (pred_prob > 0.5).astype(np.float32)
    
    # Clip images to [0, 1] for display
    img_A = np.clip(img_A, 0, 1)
    img_B = np.clip(img_B, 0, 1)
    
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_A)
    plt.title("Time 1")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_B)
    plt.title("Time 2")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(label, cmap='gray', vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    plt.title("Prediction")
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
