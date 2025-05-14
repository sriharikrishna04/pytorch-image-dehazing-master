import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os

def compare_images(hazy_path, dehazed_path, gt_path=None, save_path=None):
    """
    Display hazy, dehazed, and ground truth images side by side.
    
    Args:
        hazy_path: Path to the hazy input image
        dehazed_path: Path to the dehazed output image
        gt_path: Path to the ground truth (clear) image (optional)
        save_path: Path to save the comparison image (optional)
    """
    # Check if files exist
    if not os.path.exists(hazy_path):
        raise FileNotFoundError(f"Hazy image not found: {hazy_path}")
    if not os.path.exists(dehazed_path):
        raise FileNotFoundError(f"Dehazed image not found: {dehazed_path}")
    
    # Open images
    hazy_img = Image.open(hazy_path)
    dehazed_img = Image.open(dehazed_path)
    
    # Determine number of subplots
    if gt_path and os.path.exists(gt_path):
        gt_img = Image.open(gt_path)
        num_images = 3
        images = [hazy_img, dehazed_img, gt_img]
        titles = ["Hazy", "Dehazed", "Ground Truth"]
    else:
        num_images = 2
        images = [hazy_img, dehazed_img]
        titles = ["Hazy", "Dehazed"]
        if gt_path:
            print(f"Warning: Ground truth image not found: {gt_path}")
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    # Handle single axis case
    if num_images == 1:
        axes = [axes]
    
    # Loop through images and plot them with titles
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")  # Hide axes
    
    # Add a main title with the filename
    plt.suptitle(f"Image Comparison: {os.path.basename(hazy_path)}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f"Comparison saved to {save_path}")
    
    # Show the figure
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare hazy, dehazed, and ground truth images')
    
    # Required arguments
    parser.add_argument('--hazy', type=str, required=True, help='Path to hazy input image')
    parser.add_argument('--dehazed', type=str, required=True, help='Path to dehazed output image')
    
    # Optional arguments
    parser.add_argument('--gt', type=str, help='Path to ground truth (clear) image')
    parser.add_argument('--save', type=str, help='Path to save the comparison image')
    
    # Batch mode (directory-based)
    parser.add_argument('--batch', action='store_true', help='Process all images in the directories')
    parser.add_argument('--hazy_dir', type=str, help='Directory containing hazy images')
    parser.add_argument('--dehazed_dir', type=str, help='Directory containing dehazed images')
    parser.add_argument('--gt_dir', type=str, help='Directory containing ground truth images')
    parser.add_argument('--save_dir', type=str, help='Directory to save comparison images')
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch and args.hazy_dir and args.dehazed_dir:
        if not os.path.isdir(args.hazy_dir):
            print(f"Error: Hazy directory not found: {args.hazy_dir}")
            return
        if not os.path.isdir(args.dehazed_dir):
            print(f"Error: Dehazed directory not found: {args.dehazed_dir}")
            return
        
        # Create save directory if needed
        if args.save_dir and not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        # Get list of hazy images
        hazy_files = [f for f in os.listdir(args.hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not hazy_files:
            print(f"No images found in hazy directory: {args.hazy_dir}")
            return
            
        print(f"Found {len(hazy_files)} images to compare")
        
        for hazy_file in hazy_files:
            # For O-HAZE dataset, the GT filename is usually the base part without hazing parameters
            base_name = '_'.join(hazy_file.split('_')[:1]) + '.png'  # Assumes GT is PNG
            
            hazy_path = os.path.join(args.hazy_dir, hazy_file)
            dehazed_path = os.path.join(args.dehazed_dir, hazy_file)
            
            # Skip if dehazed image doesn't exist
            if not os.path.exists(dehazed_path):
                print(f"Warning: Dehazed image not found for {hazy_file}, skipping")
                continue
            
            # Set ground truth path if available
            gt_path = None
            if args.gt_dir:
                gt_path = os.path.join(args.gt_dir, base_name)
                # Try with same extension as hazy if PNG not found
                if not os.path.exists(gt_path):
                    alt_gt_path = os.path.join(args.gt_dir, base_name.rsplit('.', 1)[0] + '.' + hazy_file.rsplit('.', 1)[1])
                    if os.path.exists(alt_gt_path):
                        gt_path = alt_gt_path
            
            # Set save path if requested
            save_path = None
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f"comparison_{hazy_file}")
            
            try:
                print(f"\nComparing images for {hazy_file}...")
                compare_images(hazy_path, dehazed_path, gt_path, save_path)
            except Exception as e:
                print(f"Error processing {hazy_file}: {e}")
    
    # Single image mode
    else:
        try:
            compare_images(args.hazy, args.dehazed, args.gt, args.save)
        except Exception as e:
            print(f"Error: {e}")
            parser.print_help()

if __name__ == "__main__":
    main() 