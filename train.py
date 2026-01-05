import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.siamese_net import SiameseUNet
from utils.dataset import LEVIRCDDataset
from utils.metrics import calculate_metrics, AverageMeter
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Loader
    train_dataset = LEVIRCDDataset(args.data_dir, split='train')
    val_dataset = LEVIRCDDataset(args.data_dir, split='val')
    
    if len(train_dataset) == 0:
        print(f"No training data found in {args.data_dir}/train. Please check dataset path.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for safe windows execution
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = SiameseUNet().to(device)
    
    # Loss & Optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(img_A, img_B)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), img_A.size(0))
            pbar.set_postfix({'loss': f"{train_loss.avg:.4f}"})
        
        scheduler.step()
            
        # Validation
        if len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Epoch {epoch+1} Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_path = os.path.join('models', 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        else:
            # Save anyway if no validation set
            save_path = os.path.join('models', 'latest_model.pth')
            torch.save(model.state_dict(), save_path)

def validate(model, loader, device):
    model.eval()
    val_loss = AverageMeter()
    metrics_sum = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            img_A = batch['image_A'].to(device)
            img_B = batch['image_B'].to(device)
            label = batch['label'].to(device)
            
            output = model(img_A, img_B)
            loss = criterion(output, label)
            val_loss.update(loss.item(), img_A.size(0))
            
            batch_metrics = calculate_metrics(output, label)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
                
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
    avg_metrics['loss'] = val_loss.avg
    return avg_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    train(args)
