import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

DATA_DIR = "data"
CLASSES = ["white", "yellow", "red", "orange", "green", "blue"]
# approximate RGB values for Rubik's cube colors
COLORS = {
    "white": (255, 255, 255),
    "yellow": (255, 213, 0),
    "red": (185, 0, 0),
    "orange": (255, 89, 0),
    "green": (0, 155, 72),
    "blue": (0, 69, 173)
}

def create_noisy_color_patch(color_name, size=(64, 64)):
    base_color = COLORS[color_name]

    # Create base image
    img = Image.new("RGB", size, base_color)

    # Add noise
    np_img = np.array(img, dtype=np.int16)
    noise = np.random.randint(-30, 30, np_img.shape)
    np_img = np_img + noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    img = Image.fromarray(np_img)

    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.5, 1.5)
    img = enhancer.enhance(factor)

    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(0.7, 1.3)
    img = enhancer.enhance(factor)

    # Random rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=base_color) # fill with base color to avoid black borders

    # Resize to target input size (e.g., 32x32 for the model)
    img = img.resize((32, 32))

    return img

def generate_dataset(samples_per_class=500):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for label_idx, color_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, color_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for i in range(samples_per_class):
            img = create_noisy_color_patch(color_name)
            filename = os.path.join(class_dir, f"{i}.jpg")
            img.save(filename)

    print(f"Generated {samples_per_class * len(CLASSES)} images in {DATA_DIR}")

if __name__ == "__main__":
    generate_dataset()
