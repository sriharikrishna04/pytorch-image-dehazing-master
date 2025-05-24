import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import argparse

def enhance_dehazed_image(image_path, clahe_clip_limit, clahe_tile_grid_size, color_enhance_factor):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Color enhancement
    pil_image = Image.fromarray(enhanced_image)
    enhancer = ImageEnhance.Color(pil_image)
    color_enhanced = enhancer.enhance(color_enhance_factor)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes[0].imshow(image)
    axes[0].set_title("Original Dehazed Image")
    axes[0].axis("off")

    axes[1].imshow(color_enhanced)
    axes[1].set_title("Color Enhanced Image")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    return color_enhanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a dehazed image using CLAHE and color boost.")
    parser.add_argument('--image', type=str, required=True, help='Path to the dehazed image')
    parser.add_argument('--clip', type=float, default=3.0, help='CLAHE clip limit')
    parser.add_argument('--grid', nargs=2, type=int, default=[8, 8], help='CLAHE tile grid size as two integers')
    parser.add_argument('--color', type=float, default=1.5, help='Color enhancement factor')

    args = parser.parse_args()

    enhance_dehazed_image(
        image_path=args.image,
        clahe_clip_limit=args.clip,
        clahe_tile_grid_size=tuple(args.grid),
        color_enhance_factor=args.color
    )
