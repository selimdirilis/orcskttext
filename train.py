import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OCRDataset
from model import CRNN

# 🔤 Karakter kümesi
CHARSET = list("0123456789.SKTY") + ['<BLANK>', '<UNK>']
char_to_idx = {char: idx for idx, char in enumerate(CHARSET)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# 📐 Ayarlar
IMG_DIR = "images"
LABEL_FILE = "labels.txt"
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = (200, 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📊 Dataset
train_dataset = OCRDataset(IMG_DIR, LABEL_FILE, char_to_idx, img_size=IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

# 🧠 Model + Loss + Optimizasyon
model = CRNN(img_height=IMG_SIZE[1] // 8, num_classes=len(CHARSET)).to(device)
criterion = nn.CTCLoss(blank=char_to_idx['<BLANK>'], zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 🧮 Yardımcı: padding-free label lengths
def collate_batch(batch):
    images, targets = zip(*batch)
    input_lengths = torch.full(size=(len(images),), fill_value=IMG_SIZE[0] // 8, dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = [item for sublist in targets for item in sublist]
    return (
        torch.stack(images).to(device),
        torch.tensor(targets_flat, dtype=torch.long).to(device),
        input_lengths.to(device),
        target_lengths.to(device)
    )


# 🚂 Eğitim döngüsü
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs, targets, input_lengths, target_lengths = collate_batch(batch)
        optimizer.zero_grad()
        outputs = model(inputs)  # (B, W, num_classes)
        outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (W, B, C)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[{epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")

# 💾 Modeli kaydet
torch.save(model.state_dict(), "skt_crnn.pth")
print("✅ Eğitim tamamlandı ve model kaydedildi.")
