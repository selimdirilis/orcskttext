import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OCRDataset(Dataset):
    def __init__(self, img_dir, label_file, char_to_idx, img_size=(200, 80)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.char_to_idx = char_to_idx
        self.data = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                filename, label = line.strip().split(",", 1)
                self.data.append((filename, label))

        self.transform = transforms.Compose([
            transforms.Resize((80, 200)),  # (Yükseklik, Genişlik)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # gri görüntü için normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("L")  # Gri tonlama
        image = self.transform(image)

        # karakterleri indexlere çeviriyoruz
        label_indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in label]
        return image, label_indices
