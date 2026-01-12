import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = "data/images"
LABEL_FILE = "data/dataset.json"
NUM_SAMPLES = 500

COLORS = [
    ("White", (255, 255, 255), 0),
    ("Yellow", (255, 255, 0), 1),
    ("Green", (0, 155, 72), 2),
    ("Blue", (0, 69, 173), 3),
    ("Red", (185, 0, 0), 4),
    ("Orange", (255, 89, 0), 5)
]

def add_noise(image):
    arr = np.array(image)
    noise = np.random.randint(-20, 20, arr.shape)
    arr = arr.astype(np.int16) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_sample(sample_id):
    img_size = 128
    cell_size = img_size // 3
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    sticker_colors = []

    for r in range(3):
        for c in range(3):
            color_name, rgb, color_id = random.choice(COLORS)
            sticker_colors.append(color_id)
            jitter = lambda x: max(0, min(255, x + random.randint(-15, 15)))
            rgb_jittered = (jitter(rgb[0]), jitter(rgb[1]), jitter(rgb[2]))
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=rgb_jittered, outline=(0, 0, 0), width=2)
            gap = 2
            draw.rectangle([x0+gap, y0+gap, x1-gap, y1-gap], fill=rgb_jittered)

    img = add_noise(img)
    filename = f"sample_{sample_id}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    return {
        "id": filename,
        "labels": sticker_colors
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    dataset = []
    print(f"Generating {NUM_SAMPLES} samples...")
    for i in range(NUM_SAMPLES):
        sample = generate_sample(i)
        dataset.append(sample)
    with open(LABEL_FILE, "w") as f:
        json.dump(dataset, f)
    print("Done.")

if __name__ == "__main__":
    main()
