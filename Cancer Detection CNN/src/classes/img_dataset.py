# custom dataset class for image test data
from torch.utils.data import Dataset
from PIL import Image
import os


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_id = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_id[idx])
        image = Image.open(img_path).convert("RGB")
        label = 0
        if self.transform:
            image = self.transform(image)
        return image, label, self.img_id[idx]
