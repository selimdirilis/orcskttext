import os
import random
from PIL import Image, ImageDraw, ImageFont

# Ayarlar
NUM_IMAGES = 300
IMAGE_SIZE = (200, 80)
SAVE_DIR = "images"
LABELS_PATH = "labels.txt"

os.makedirs(SAVE_DIR, exist_ok=True)

# Sistemden tek font kullan
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def random_skt_text():
    mode = random.choice(["date", "skt", "tsy", "none"])
    if mode == "date":
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(2023, 2030)
        return f"{day:02}.{month:02}.{year}"
    elif mode == "skt":
        return "SKT"
    elif mode == "tsy":
        return "TSYY"
    else:
        return ""

def draw_text_image(text):
    img = Image.new("RGB", IMAGE_SIZE, color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 32)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((IMAGE_SIZE[0] - text_width) / 2, (IMAGE_SIZE[1] - text_height) / 2)
    draw.text(position, text, font=font, fill="black")
    return img

with open(LABELS_PATH, "w", encoding="utf-8") as label_file:
    for i in range(1, NUM_IMAGES + 1):
        label = random_skt_text()
        img = draw_text_image(label)
        filename = f"skt_{i:03}.jpg"
        img.save(os.path.join(SAVE_DIR, filename))
        label_file.write(f"{filename},{label}\n")
