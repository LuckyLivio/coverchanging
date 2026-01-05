import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.siamese_net import SiameseUNet
from utils.dataset import LEVIRCDDataset
from utils.visualize import visualize_result
from tqdm import tqdm

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = SiameseUNet().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model path {args.model_path} not found. Please train the model first.")
        return
        
    model.eval()
    
    # Dataset
    test_dataset = LEVIRCDDataset(args.data_dir, split='test')
    if len(test_dataset) == 0:
        print(f"No test data found in {args.data_dir}/test.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to {save_dir}...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)
            name = batch['name'][0]
            
            output = model(img_A, img_B)
            
            # Visualize first N images
            if i < args.num_viz:
                save_path = os.path.join(save_dir, f"pred_{name[:-4]}.png")
                visualize_result(img_A[0], img_B[0], label[0], output[0], save_path=save_path)
            else:
                break
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    parser.add_argument('--num_viz', type=int, default=20, help='Number of images to visualize')
    args = parser.parse_args()
    
    predict(args)
