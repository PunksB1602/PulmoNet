import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.labels = [
            'Atelectasis', 
            'Cardiomegaly', 
            'Consolidation', 
            'Edema', 
            'Pleural Effusion'
        ]
        
        # Simple cleaning strategy (U-Ones): fill NaNs with 0 and map -1 (uncertain) -> 1.
        self.data[self.labels] = self.data[self.labels].fillna(0)
        self.data[self.labels] = self.data[self.labels].replace(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # CSV typically stores paths like 'CheXpert-v1.0-small/train/...'.
        img_path_in_csv = self.data.iloc[idx, 0]

        img_name = os.path.join(self.root_dir, img_path_in_csv)
        
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = torch.tensor(self.data.iloc[idx][self.labels].values.astype(float), dtype=torch.float32)
        
        return image, labels

def get_transforms(mode='train'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])