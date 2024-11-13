from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np
import torch
import matplotlib.pyplot as plt 
from PIL import Image
import cv2

def calculate_complexity(img):
    """
    Calculate image complexity using edge density and structural features
    """
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Count significant structural elements
    num_edges = np.sum(edges > 0)
    edge_density = num_edges / edges.size
    
    # Find contours to count distinct objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = len(contours)
    
    return {
        'edge_density': edge_density,
        'num_objects': num_objects,
        'complexity_score': edge_density * (1 + np.log(1 + num_objects))
    }
def analyze_differences(diff_map, threshold=0.1):
    """
    Analyze the differences between images using connected components
    """
    # If RGB, convert to grayscale by taking mean across channels
    if len(diff_map.shape) == 3:
        diff_map = np.mean(diff_map, axis=2)
    
    # Threshold the difference map
    diff_binary = (diff_map > threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_binary)
    
    # Calculate area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    
    # Weight larger connected differences more heavily
    weighted_diff = 0
    if len(areas) > 0:
        avg_area = np.mean(areas)
        weighted_diff = np.sum(areas * (areas > avg_area)) / diff_binary.size
    
    return {
        'num_components': num_labels - 1,  # Subtract background
        'weighted_diff': weighted_diff,
        'component_areas': areas
    }
def create_foreground_mask(img, threshold=250):
    """
    Create a mask for non-background (non-white) pixels
    """
    if len(img.shape) == 3:
        # For RGB images, consider a pixel as background if all channels are near white
        mask = np.all(img > threshold, axis=2)
    else:
        mask = img > threshold
    # Invert mask (True for foreground pixels)
    return ~mask

def compute_image_similarity_with_components(img1, img2):
    """
    Compute improved similarity metrics considering complexity and structural differences,
    ignoring white background
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy().squeeze()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy().squeeze()
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Create foreground masks
    mask1 = create_foreground_mask(img1)
    mask2 = create_foreground_mask(img2)
    # Combined mask (where either image has foreground)
    combined_mask = mask1 | mask2
    
    # Expand mask to 3D if images are RGB
    if len(img1.shape) == 3:
        combined_mask = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
    
    # Calculate SSIM only on foreground pixels
    ssim_score, ssim_map = ssim(img1, 
                               img2,
                               data_range=255,
                               multichannel=True,
                               channel_axis=-1,
                               full=True)
    
    # Apply mask to SSIM map
    ssim_map = ssim_map * combined_mask
    # Recalculate SSIM score for foreground only
    normalized_ssim = 1 - np.mean(ssim_map[combined_mask])
    
    # Calculate absolute differences (foreground only)
    diff_map = np.abs(img1.astype('float') - img2.astype('float')) / 255.0
    diff_map = diff_map * combined_mask
    abs_diff = np.mean(diff_map[combined_mask]) if np.any(combined_mask) else 0
    
    # Calculate MSE (foreground only)
    mse_map = (img1.astype('float') - img2.astype('float'))**2 / (255.0**2)
    mse_map = mse_map * combined_mask
    mse = np.mean(mse_map[combined_mask]) if np.any(combined_mask) else 0
    rmse = np.sqrt(mse)
    
    # Calculate complexity for both images
    complexity1 = calculate_complexity(img1)
    complexity2 = calculate_complexity(img2)
    complexity_factor = max(complexity1['complexity_score'], 
                          complexity2['complexity_score'])
    
    # Analyze differences (using masked diff_map)
    diff_analysis = analyze_differences(diff_map)
    
    # Calculate weighted score
    structural_score = diff_analysis['weighted_diff'] * (1 + complexity_factor)
    
    # Combined score with adjusted weights
    combined_score = (
        0.125 * rmse +
        0.125 * normalized_ssim +
        0.75 * structural_score  # Give more weight to structural differences
    )
    
    return {
        'rmse': rmse,
        'ssim': normalized_ssim,
        'abs_diff': abs_diff,
        'combined_score': combined_score,
        'ssim_map': ssim_map,
        'mse_map': mse_map,
        'diff_map': diff_map,
        'complexity_factor': complexity_factor,
        'num_diff_components': diff_analysis['num_components'],
        'structural_score': structural_score,
        'foreground_mask': combined_mask  # Added for visualization
    }

def visualize_similarity(img1, img2, similarity_results):
    """
    Visualize the similarity maps for RGB images in color
    """
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(2, 3, figure=fig)
    
    # Original Images (top row)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    
    # Similarity Maps (bottom row)
    ax4 = fig.add_subplot(grid[1, 0])
    ax5 = fig.add_subplot(grid[1, 1])
    ax6 = fig.add_subplot(grid[1, 2])
    
    # Show original RGB images
    ax1.imshow(img1)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title('Generated Image')
    ax2.axis('off')
    
    # Show difference image in RGB
    im_diff = ax3.imshow(similarity_results['diff_map'])
    ax3.set_title(f'Absolute Map\nAbsolute: {similarity_results["abs_diff"]:.4f}')
    plt.colorbar(im_diff, ax=ax3)
    ax3.axis('off')
    
    # Show SSIM map in RGB
    im_ssim = ax4.imshow(similarity_results['ssim_map'])
    ax4.set_title(f'SSIM Map\nScore: {similarity_results["ssim"]:.4f}')
    plt.colorbar(im_ssim, ax=ax4)
    ax4.axis('off')
    
    # Show MSE map in RGB
    im_mse = ax5.imshow(similarity_results['mse_map'])
    ax5.set_title(f'MSE Map\nRMSE: {similarity_results["rmse"]:.4f}\nComplexity: {similarity_results["complexity_factor"]:.4f}')
    plt.colorbar(im_mse, ax=ax5)
    ax5.axis('off')
    
    # Combined visualization in RGB
    combined_map = 0.3 * similarity_results['mse_map'] + 0.3 * similarity_results['ssim_map'] + 0.4 * similarity_results['diff_map']
    im_combined = ax6.imshow(combined_map)
    ax6.set_title(f'Combined Error Map\nScore: {similarity_results["combined_score"]:.4f}\nStructural Score: {similarity_results["structural_score"]:.4f}')
    plt.colorbar(im_combined, ax=ax6)
    ax6.axis('off')
    
    plt.tight_layout()
    return fig


def preprocess_images(img1, img2):
    """
    Preprocess images for comparison with better size handling
    """
    if isinstance(img1, str):
        with Image.open(img1) as img:
            img1 = img.convert('RGB').copy()  # .copy() to keep the image after context exits
    if isinstance(img2, str):
        with Image.open(img2) as img:
            img2 = img.convert('RGB').copy()
        
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate target size (use max dimensions to preserve details)
    target_h = max(h1, h2)
    target_w = max(w1, w2)
    
    # Resize images while maintaining aspect ratio
    def resize_with_padding(image, target_height, target_width):
        h, w = image.shape[:2]
        scale = min(target_height/h, target_width/w)
        # print(h, w)
        # print(target_height, target_width)
        # New dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        # print(new_h, new_w)
        # Resize using cv2 for better quality
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Calculate padding
        pad_h = target_height - new_h
        pad_w = target_width - new_w
        
        # Add padding
        pad_top = pad_h//2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w//2
        pad_right = pad_w - pad_left
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]  # White padding
        )
        
        return padded
    
    # Resize both images
    img1_processed = resize_with_padding(img1, target_h, target_w)
    img2_processed = resize_with_padding(img2, target_h, target_w)
    
    return img1_processed, img2_processed

def compare_images(image_path1, image_path2, visualize = False):
    # Load and preprocess images
    img1_array, img2_array = preprocess_images(image_path1, image_path2)
    
    # print(f"Original dimensions - Original: {img1_array.shape}, Generated: {img2_array.shape}")
    
    # Compute similarity
    similarity = compute_image_similarity_with_components(img1_array, img2_array)
    if visualize:
        fig = visualize_similarity(img1_array, img2_array, similarity)
    
        return fig, similarity
    else:
        return similarity



if __name__ == "__main__":
    # Compare two images
    original_path = '/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ/tikz_results_20241105_220928/images/example_0/combination_14.png'
    generated_path = '/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ/tikz_results_20241105_220928/generated/example_0/generated_14.png'
    fig, metrics = compare_images(original_path,generated_path , visualize = True)
    plt.show()

    # Save visualization
    fig.savefig('comparison.png', dpi=300, bbox_inches='tight')
