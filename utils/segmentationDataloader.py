import glob
from pathlib import Path
from torchvision import transforms
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset


class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.img_files = glob.glob(str(Path(folder_path)/'images'/'*.png'))
        self.mask_files = glob.glob(str(Path(folder_path)/'masks'/'*.png'))
        self.transforms_data = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transforms_mask = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])
        self.transforms_both = transforms.Compose([
            transforms.Resize(256)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = np.array(Image.open(img_path).convert('RGB'))
            mask = np.array(Image.open(mask_path))
            if self.transforms_data:
                data = self.transforms_data(self.transforms_both(Image.fromarray(data)))
                mask = self.transforms_mask(self.transforms_both(Image.fromarray(mask)))
                #data = self.transforms_data(Image.fromarray(data))
                #mask = self.transforms_mask(Image.fromarray(mask))
            return (data,mask)