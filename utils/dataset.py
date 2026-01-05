import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class LEVIRCDDataset(Dataset):
    """
    Dataset class for LEVIR-CD or similar change detection datasets.
    Expected structure:
    root_dir/
        train/
            A/ (Time 1 images)
            B/ (Time 2 images)
            label/ (Binary change masks)
        val/
            ...
        test/
            ...
    """
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        self.split = split
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        
        # Define paths
        self.dir_A = os.path.join(root_dir, split, 'A')
        self.dir_B = os.path.join(root_dir, split, 'B')
        self.dir_label = os.path.join(root_dir, split, 'label')
        
        # Check if directories exist
        if os.path.exists(self.dir_A) and os.path.exists(self.dir_B) and os.path.exists(self.dir_label):
            self.image_names = sorted(os.listdir(self.dir_A))
            # Verify consistency
            b_names = set(os.listdir(self.dir_B))
            l_names = set(os.listdir(self.dir_label))
            self.image_names = [x for x in self.image_names if x in b_names and x in l_names]
        else:
            self.image_names = []
            print(f"Warning: Data directories for split '{split}' not found in {root_dir}")

        # Default transforms
        self.t_img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.t_label = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path_A = os.path.join(self.dir_A, name)
        path_B = os.path.join(self.dir_B, name)
        path_label = os.path.join(self.dir_label, name)

        # Load images
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        label = Image.open(path_label).convert('L') # Binary mask

        # Apply transforms
        # Note: In a real advanced setting, we would handle random crop/flip jointly.
        # Here we stick to deterministic resize for simplicity in this demo.
        img_A = self.t_img(img_A)
        img_B = self.t_img(img_B)
        label = self.t_label(label)
        
        # Binarize label (0 or 1)
        label = (label > 0.5).float()

        return {'image_A': img_A, 'image_B': img_B, 'label': label, 'name': name}
