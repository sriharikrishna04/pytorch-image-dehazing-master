import argparse
from math import log10, sqrt 
import cv2 
import numpy as np 
from skimage.metrics import structural_similarity as ssim
import os

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def ssim_compare(img1, img2):
    # Convert to grayscale if not already
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    
    # Resize to match dimensions if needed
    if img1_gray.shape != img2_gray.shape:
        img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    
    ssim_score, _ = ssim(img1_gray, img2_gray, full=True)
    return ssim_score

def compute_entropy(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
        
    hist, _ = np.histogram(image_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize histogram
    hist = hist[hist > 0]  # Remove zeros to avoid log issues
    return -np.sum(hist * np.log2(hist))

def evaluate_image(hazy_path, gt_path, dehazed_path, verbose=True):
    """Evaluate a single set of images and return metrics"""
    # Load images
    hazy_img = cv2.imread(hazy_path)
    gt_img = cv2.imread(gt_path)
    dehazed_img = cv2.imread(dehazed_path)
    
    if hazy_img is None:
        raise ValueError(f"Error: Could not load hazy image from {hazy_path}")
    if gt_img is None:
        raise ValueError(f"Error: Could not load ground truth image from {gt_path}")
    if dehazed_img is None:
        raise ValueError(f"Error: Could not load dehazed image from {dehazed_path}")
    
    # Resize to match dimensions if needed
    if gt_img.shape != dehazed_img.shape:
        dehazed_img = cv2.resize(dehazed_img, (gt_img.shape[1], gt_img.shape[0]))
    if gt_img.shape != hazy_img.shape:
        hazy_img = cv2.resize(hazy_img, (gt_img.shape[1], gt_img.shape[0]))
    
    # Calculate metrics
    psnr_value = PSNR(gt_img, dehazed_img)
    ssim_value = ssim_compare(gt_img, dehazed_img)
    
    entropy_gt = compute_entropy(gt_img)
    entropy_dehazed = compute_entropy(dehazed_img)
    entropy_hazy = compute_entropy(hazy_img)
    
    # Print results
    if verbose:
        print(f"\nEvaluating: {os.path.basename(hazy_path)}")
        print(f"PSNR value: {psnr_value:.2f} dB")
        print(f"SSIM value: {ssim_value:.4f}")
        print(f"Entropy of Ground Truth Image: {entropy_gt:.4f}")
        print(f"Entropy of Dehazed Image: {entropy_dehazed:.4f}")
        print(f"Entropy of Hazy Image: {entropy_hazy:.4f}")
        
    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'entropy_gt': entropy_gt,
        'entropy_dehazed': entropy_dehazed,
        'entropy_hazy': entropy_hazy
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate image dehazing results')
    
    # Single image evaluation
    parser.add_argument('--hazy', type=str, help='Path to hazy input image')
    parser.add_argument('--gt', type=str, help='Path to ground truth clear image')
    parser.add_argument('--dehazed', type=str, help='Path to dehazed output image')
    
    # Directory-based evaluation
    parser.add_argument('--hazy_dir', type=str, help='Directory containing hazy images')
    parser.add_argument('--gt_dir', type=str, help='Directory containing ground truth images')
    parser.add_argument('--dehazed_dir', type=str, help='Directory containing dehazed images')
    
    # Output options
    parser.add_argument('--save_csv', type=str, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    all_results = []
    
    # Evaluate single images
    if args.hazy and args.gt and args.dehazed:
        results = evaluate_image(args.hazy, args.gt, args.dehazed)
        all_results.append({
            'filename': os.path.basename(args.hazy),
            **results
        })
    
    # Evaluate directories
    elif args.hazy_dir and args.gt_dir and args.dehazed_dir:
        if not os.path.isdir(args.hazy_dir) or not os.path.isdir(args.gt_dir) or not os.path.isdir(args.dehazed_dir):
            print("Error: One or more directories do not exist")
            return
            
        # Get list of hazy images
        hazy_files = [f for f in os.listdir(args.hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not hazy_files:
            print(f"No images found in hazy directory: {args.hazy_dir}")
            return
            
        print(f"Found {len(hazy_files)} images to evaluate")
        
        # Track overall metrics
        total_psnr = 0
        total_ssim = 0
        success_count = 0
        
        for hazy_file in hazy_files:
            # For O-HAZE dataset, the GT filename is usually the base part without hazing parameters
            base_name = '_'.join(hazy_file.split('_')[:1]) + '.png'  # Assumes GT is PNG, modify if needed
            
            hazy_path = os.path.join(args.hazy_dir, hazy_file)
            gt_path = os.path.join(args.gt_dir, base_name)
            dehazed_path = os.path.join(args.dehazed_dir, hazy_file)
            
            # Check if files exist
            if not os.path.exists(gt_path):
                # Try with same extension as hazy if PNG not found
                alt_gt_path = os.path.join(args.gt_dir, base_name.rsplit('.', 1)[0] + '.' + hazy_file.rsplit('.', 1)[1])
                if os.path.exists(alt_gt_path):
                    gt_path = alt_gt_path
                else:
                    print(f"Warning: Ground truth image not found for {hazy_file}")
                    continue
                    
            if not os.path.exists(dehazed_path):
                print(f"Warning: Dehazed image not found for {hazy_file}")
                continue
                
            try:
                results = evaluate_image(hazy_path, gt_path, dehazed_path)
                all_results.append({
                    'filename': hazy_file,
                    **results
                })
                
                total_psnr += results['psnr']
                total_ssim += results['ssim']
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {hazy_file}: {e}")
        
        if success_count > 0:
            print("\n=== Overall Results ===")
            print(f"Average PSNR: {total_psnr / success_count:.2f} dB")
            print(f"Average SSIM: {total_ssim / success_count:.4f}")
            print(f"Successfully processed {success_count} out of {len(hazy_files)} images")
    
    else:
        print("Error: You must provide either individual image paths or directories")
        parser.print_help()
        return
        
    # Save results to CSV if requested
    if args.save_csv and all_results:
        import csv
        with open(args.save_csv, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'psnr', 'ssim', 'entropy_gt', 'entropy_dehazed', 'entropy_hazy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
            
        print(f"Results saved to {args.save_csv}")

if __name__ == "__main__":
    main() 