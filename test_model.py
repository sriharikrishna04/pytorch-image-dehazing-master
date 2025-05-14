import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
import cv2
from torch.autograd import Variable
from model import Generator
from utils import rgb_to_tensor, tensor_to_rgb, PSNR
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")

def get_args():
    parser = argparse.ArgumentParser(description='test-model-dehazing')
    parser.add_argument('--model', required=True, help='path to trained model')
    parser.add_argument('--hazy_dir', required=True, help='path to hazy images directory')
    parser.add_argument('--gt_dir', required=True, help='path to ground truth images directory')
    parser.add_argument('--output_dir', default='test_outputs', help='directory to save dehazed images')
    parser.add_argument('--num_images', type=int, default=50, help='number of random images to test')
    parser.add_argument('--gpu', type=int, required=True, help='gpu index')
    return parser.parse_args()

def verify_paths(hazy_dir, gt_dir):
    """Pair hazy and ground truth images based on prefix matching."""
    if not os.path.exists(hazy_dir):
        raise ValueError(f"Hazy images directory does not exist: {hazy_dir}")
    if not os.path.exists(gt_dir):
        raise ValueError(f"Ground truth directory does not exist: {gt_dir}")

    hazy_images = [f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_images = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not hazy_images:
        raise ValueError(f"No images found in hazy directory: {hazy_dir}")
    if not gt_images:
        raise ValueError(f"No images found in ground truth directory: {gt_dir}")

    print(f"Found {len(hazy_images)} images in hazy directory")
    print(f"Found {len(gt_images)} images in ground truth directory")

    image_pairs = []
    gt_basenames = {os.path.splitext(gt)[0]: gt for gt in gt_images}

    for hazy_img in hazy_images:
        hazy_prefix = os.path.splitext(hazy_img)[0].split('_')[0]
        if hazy_prefix in gt_basenames:
            image_pairs.append((hazy_img, gt_basenames[hazy_prefix]))

    if not image_pairs:
        raise ValueError("No matching pairs found between hazy and ground truth images")

    print(f"Found {len(image_pairs)} matching pairs of images")
    return image_pairs

def compute_ssim(img1, img2):
    # Convert to grayscale if images are in color
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure both images have the same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    ssim_score, _ = ssim(img1, img2, full=True)
    return ssim_score

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model file does not exist: {model_path}")
    
    model = Generator()
    model = model.to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['netG'])
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def process_image(image_path, model, device):
    try:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        scale = 32
        image = image.resize((width // scale * scale, height // scale * scale))
        
        with torch.no_grad():
            image = rgb_to_tensor(image)
            image = image.unsqueeze(0)
            image = Variable(image.to(device))
            output = model(image)
        
        output = tensor_to_rgb(output)
        out = Image.fromarray(np.uint8(output), mode='RGB')
        out = out.resize((width, height), resample=Image.BICUBIC)
        return out
    except Exception as e:
        raise RuntimeError(f"Error processing image {image_path}: {str(e)}")

def calculate_metrics(dehazed_img, gt_img, hazy_img):
    try:
        # Convert PIL images to numpy arrays
        dehazed_np = np.array(dehazed_img)
        gt_np = np.array(gt_img)
        hazy_np = np.array(hazy_img)
        
        # Convert RGB to BGR for OpenCV
        dehazed_np = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
        gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
        hazy_np = cv2.cvtColor(hazy_np, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        psnr = PSNR(gt_np, dehazed_np)
        ssim_value = compute_ssim(gt_np, dehazed_np)
        entropy_gt = compute_entropy(gt_np)
        entropy_dehazed = compute_entropy(dehazed_np)
        entropy_hazy = compute_entropy(hazy_np)
        
        return psnr, ssim_value, entropy_gt, entropy_dehazed, entropy_hazy
    except Exception as e:
        raise RuntimeError(f"Error calculating metrics: {str(e)}")

def compute_entropy(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize histogram
    hist = hist[hist > 0]  # Remove zeros to avoid log issues
    return -np.sum(hist * np.log2(hist))

def plot_metrics(metrics_df, output_dir):
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot PSNR distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=metrics_df, x='PSNR', bins=20)
    plt.title('PSNR Distribution')
    plt.savefig(os.path.join(metrics_dir, 'psnr_distribution.png'))
    plt.close()
    
    # Plot SSIM distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=metrics_df, x='SSIM', bins=20)
    plt.title('SSIM Distribution')
    plt.savefig(os.path.join(metrics_dir, 'ssim_distribution.png'))
    plt.close()
    
    # Plot PSNR vs SSIM scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=metrics_df, x='PSNR', y='SSIM')
    plt.title('PSNR vs SSIM')
    plt.savefig(os.path.join(metrics_dir, 'psnr_vs_ssim.png'))
    plt.close()
    
    # Plot Entropy comparison
    plt.figure(figsize=(10, 6))
    entropy_data = pd.melt(metrics_df[['Hazy_Image', 'Entropy_GT', 'Entropy_Dehazed', 'Entropy_Hazy']], 
                          id_vars=['Hazy_Image'], 
                          value_vars=['Entropy_GT', 'Entropy_Dehazed', 'Entropy_Hazy'],
                          var_name='Image Type', 
                          value_name='Entropy')
    sns.boxplot(data=entropy_data, x='Image Type', y='Entropy')
    plt.title('Entropy Comparison')
    plt.savefig(os.path.join(metrics_dir, 'entropy_comparison.png'))
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)

def main():
    args = get_args()
    
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Verify input paths and get image pairs
    try:
        image_pairs = verify_paths(args.hazy_dir, args.gt_dir)
    except Exception as e:
        print(f"Error verifying paths: {str(e)}")
        return
    
    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Randomly select image pairs
    selected_pairs = random.sample(image_pairs, min(args.num_images, len(image_pairs)))
    print(f"Selected {len(selected_pairs)} image pairs for processing")
    
    # Initialize metrics storage
    metrics_data = []
    successful_images = 0
    
    # Process each image pair
    for hazy_name, gt_name in selected_pairs:
        try:
            # Process hazy image
            hazy_path = os.path.join(args.hazy_dir, hazy_name)
            gt_path = os.path.join(args.gt_dir, gt_name)
            
            # Load and process images
            hazy_img = Image.open(hazy_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            dehazed_img = process_image(hazy_path, model, device)
            
            # Save dehazed image
            output_path = os.path.join(args.output_dir, f'dehazed_{hazy_name}')
            dehazed_img.save(output_path)
            
            # Calculate metrics
            psnr, ssim_value, entropy_gt, entropy_dehazed, entropy_hazy = calculate_metrics(
                dehazed_img, gt_img, hazy_img)
            
            # Store metrics
            metrics_data.append({
                'Hazy_Image': hazy_name,
                'GT_Image': gt_name,
                'PSNR': psnr,
                'SSIM': ssim_value,
                'Entropy_GT': entropy_gt,
                'Entropy_Dehazed': entropy_dehazed,
                'Entropy_Hazy': entropy_hazy
            })
            successful_images += 1
            
        except Exception as e:
            print(f"Error processing image pair ({hazy_name}, {gt_name}): {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {successful_images} out of {len(selected_pairs)} image pairs")
    
    if not metrics_data:
        print("No images were successfully processed. Please check your input paths and image formats.")
        return
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Calculate and print average metrics
    avg_psnr = metrics_df['PSNR'].mean()
    avg_ssim = metrics_df['SSIM'].mean()
    avg_entropy_gt = metrics_df['Entropy_GT'].mean()
    avg_entropy_dehazed = metrics_df['Entropy_Dehazed'].mean()
    avg_entropy_hazy = metrics_df['Entropy_Hazy'].mean()
    
    print("\nAverage Metrics:")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"Entropy (GT): {avg_entropy_gt:.2f}")
    print(f"Entropy (Dehazed): {avg_entropy_dehazed:.2f}")
    print(f"Entropy (Hazy): {avg_entropy_hazy:.2f}")
    
    # Plot and save metrics
    plot_metrics(metrics_df, args.output_dir)
    
    print(f"\nResults saved in {args.output_dir}")
    print(f"Metrics and plots saved in {os.path.join(args.output_dir, 'metrics')}")

if __name__ == '__main__':
    main() 