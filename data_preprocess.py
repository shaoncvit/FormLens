#!/usr/bin/env python3
"""
Data Preprocessing Script for FormLens
Applies various augmentations to input images including rotation, noise, perspective, and blur.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple, Optional
import random

class ImageAugmenter:
    """Class to handle various image augmentation techniques."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the augmenter with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add_gaussian_noise(self, image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
        """Add Gaussian noise to the image."""
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255)
    
    def add_salt_pepper_noise(self, image: np.ndarray, salt_prob: float = 0.1, pepper_prob: float = 0.1) -> np.ndarray:
        """Add salt and pepper noise to the image."""
        noisy_image = image.copy()
        
        # Add salt noise (white pixels)
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        
        # Add pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by a given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated_image
    
    def random_rotation(self, image: np.ndarray, min_angle: float = -15, max_angle: float = 15) -> np.ndarray:
        """Apply random rotation within specified range."""
        angle = random.uniform(min_angle, max_angle)
        return self.rotate_image(image, angle)
    
    def perspective_transform(self, image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Apply perspective transformation to simulate camera angle."""
        h, w = image.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points with perspective distortion
        offset = min(w, h) * strength
        dst_points = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
        ])
        
        # Get perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        perspective_image = cv2.warpPerspective(image, perspective_matrix, (w, h),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return perspective_image
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def motion_blur(self, image: np.ndarray, kernel_size: int = 15, angle: float = 45) -> np.ndarray:
        """Apply motion blur to the image."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel
        rotation_matrix = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply blur
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image
    
    def random_brightness_contrast(self, image: np.ndarray, brightness_range: Tuple[float, float] = (-30, 30),
                                 contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random brightness and contrast adjustment."""
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        # Apply brightness and contrast
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted_image
    
    def apply_all_augmentations(self, image: np.ndarray, 
                              rotation_range: Tuple[float, float] = (-15, 15),
                              noise_type: str = 'gaussian',
                              perspective_strength: float = 0.1,
                              blur_type: str = 'gaussian',
                              brightness_contrast: bool = True) -> dict:
        """Apply all augmentations to the input image."""
        results = {}
        
        # Original image
        results['original'] = image.copy()
        
        # Rotation
        rotated = self.random_rotation(image, rotation_range[0], rotation_range[1])
        results['rotated'] = rotated
        
        # Noise
        if noise_type == 'gaussian':
            noisy = self.add_gaussian_noise(image)
        elif noise_type == 'salt_pepper':
            noisy = self.add_salt_pepper_noise(image)
        else:
            noisy = image.copy()
        results['noisy'] = noisy
        
        # Perspective
        perspective = self.perspective_transform(image, perspective_strength)
        results['perspective'] = perspective
        
        # Blur
        if blur_type == 'gaussian':
            blurred = self.gaussian_blur(image)
        elif blur_type == 'motion':
            blurred = self.motion_blur(image)
        else:
            blurred = image.copy()
        results['blurred'] = blurred
        
        # Brightness and Contrast
        if brightness_contrast:
            adjusted = self.random_brightness_contrast(image)
            results['brightness_contrast'] = adjusted
        
        # Combined augmentations
        combined = self.random_rotation(noisy, rotation_range[0], rotation_range[1])
        combined = self.perspective_transform(combined, perspective_strength * 0.5)
        if blur_type == 'gaussian':
            combined = self.gaussian_blur(combined, (3, 3))
        results['combined'] = combined
        
        return results

def save_augmented_images(results: dict, output_dir: str, base_filename: str):
    """Save all augmented images to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for aug_type, image in results.items():
        filename = f"{base_filename}_{aug_type}.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

def main():
    """Main function to process a single image with all augmentations."""
    parser = argparse.ArgumentParser(description='Apply various augmentations to an image')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='augmented_output', help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--rotation_range', nargs=2, type=float, default=[-15, 15],
                       help='Rotation angle range (min max)')
    parser.add_argument('--noise_type', choices=['gaussian', 'salt_pepper'], default='gaussian',
                       help='Type of noise to add')
    parser.add_argument('--perspective_strength', type=float, default=0.1,
                       help='Strength of perspective transformation')
    parser.add_argument('--blur_type', choices=['gaussian', 'motion'], default='gaussian',
                       help='Type of blur to apply')
    parser.add_argument('--no_brightness_contrast', action='store_true',
                       help='Skip brightness and contrast adjustment')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image '{args.input}'!")
        return
    
    print(f"Loaded image: {args.input}")
    print(f"Image shape: {image.shape}")
    
    # Initialize augmenter
    augmenter = ImageAugmenter(seed=args.seed)
    
    # Apply all augmentations
    results = augmenter.apply_all_augmentations(
        image,
        rotation_range=tuple(args.rotation_range),
        noise_type=args.noise_type,
        perspective_strength=args.perspective_strength,
        blur_type=args.blur_type,
        brightness_contrast=not args.no_brightness_contrast
    )
    
    # Save results
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    save_augmented_images(results, args.output, base_filename)
    
    print(f"\nAll augmented images saved to: {args.output}")
    print("Augmentation types applied:")
    for aug_type in results.keys():
        print(f"  - {aug_type}")

if __name__ == "__main__":
    main()
