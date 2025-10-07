# FormLens Data Preprocessing Script

This script provides comprehensive image augmentation capabilities for preprocessing training data for the FormLens model.

## Features

The `data_preprocess.py` script includes the following augmentation techniques:

### 🔄 **Rotation Augmentation**
- Random rotation within specified angle range
- Handles image boundaries properly with white background
- Configurable rotation range (default: -15° to +15°)

### 📸 **Noise Augmentation**
- **Gaussian Noise**: Adds random Gaussian noise to simulate sensor noise
- **Salt-Pepper Noise**: Adds random white and black pixels to simulate corruption

### 🎭 **Perspective Transformation**
- Simulates camera angle variations
- Configurable distortion strength
- Useful for simulating different document capture angles

### 🌫️ **Blur Augmentation**
- **Gaussian Blur**: Smooth blur effect
- **Motion Blur**: Simulates camera shake or motion

### 🌟 **Brightness & Contrast**
- Random brightness adjustment
- Random contrast adjustment
- Helps model adapt to different lighting conditions

## Installation

```bash
# Required dependencies
pip install opencv-python numpy
```

## Usage

### Command Line Usage

```bash
# Basic usage
python data_preprocess.py --input your_image.jpg --output augmented_results

# Advanced usage with custom parameters
python data_preprocess.py \
    --input form_image.jpg \
    --output augmented_forms \
    --rotation_range -20 20 \
    --noise_type salt_pepper \
    --perspective_strength 0.15 \
    --blur_type motion \
    --seed 42

# Help
python data_preprocess.py --help
```

### Programmatic Usage

```python
from data_preprocess import ImageAugmenter
import cv2

# Load image
image = cv2.imread('your_image.jpg')

# Initialize augmenter
augmenter = ImageAugmenter(seed=42)

# Apply individual augmentations
rotated = augmenter.random_rotation(image, -15, 15)
noisy = augmenter.add_gaussian_noise(image)
perspective = augmenter.perspective_transform(image, 0.1)
blurred = augmenter.gaussian_blur(image)

# Apply all augmentations at once
results = augmenter.apply_all_augmentations(
    image,
    rotation_range=(-15, 15),
    noise_type='gaussian',
    perspective_strength=0.1,
    blur_type='gaussian',
    brightness_contrast=True
)

# Save results
for aug_type, aug_image in results.items():
    cv2.imwrite(f'output_{aug_type}.png', aug_image)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | str | Required | Input image path |
| `--output` | str | `augmented_output` | Output directory |
| `--seed` | int | None | Random seed for reproducibility |
| `--rotation_range` | float, float | `-15 15` | Rotation angle range (min max) |
| `--noise_type` | str | `gaussian` | Noise type: `gaussian` or `salt_pepper` |
| `--perspective_strength` | float | `0.1` | Perspective transformation strength |
| `--blur_type` | str | `gaussian` | Blur type: `gaussian` or `motion` |
| `--no_brightness_contrast` | flag | False | Skip brightness/contrast adjustment |

## Output

The script generates the following augmented images:

- `original`: Original input image
- `rotated`: Image with random rotation
- `noisy`: Image with added noise
- `perspective`: Image with perspective transformation
- `blurred`: Image with blur effect
- `brightness_contrast`: Image with brightness/contrast adjustment
- `combined`: Image with multiple augmentations combined

## Example Output Structure

```
augmented_output/
├── form_image_original.png
├── form_image_rotated.png
├── form_image_noisy.png
├── form_image_perspective.png
├── form_image_blurred.png
├── form_image_brightness_contrast.png
└── form_image_combined.png
```

## Running Examples

```bash
# Run example usage script
python example_usage.py

# This will create sample images and demonstrate all augmentation types
```

## Use Cases

This preprocessing script is particularly useful for:

1. **Training Data Augmentation**: Increase dataset diversity for better model generalization
2. **Robustness Testing**: Test model performance under various image conditions
3. **Data Simulation**: Simulate real-world capture conditions (camera angles, lighting, noise)
4. **Benchmark Creation**: Create challenging test cases for model evaluation

## Tips for FormLens Training

1. **Rotation Range**: Use moderate rotation (-15° to +15°) for handwritten forms
2. **Noise Type**: Gaussian noise works well for scanned documents, salt-pepper for mobile captures
3. **Perspective Strength**: Keep low (0.05-0.15) to maintain readability
4. **Blur Type**: Motion blur simulates camera shake, Gaussian blur simulates focus issues

## Contributing

Feel free to add more augmentation techniques or improve existing ones:

- Color space augmentations (HSV, LAB)
- Geometric transformations (shear, elastic)
- Advanced noise types (Poisson, speckle)
- Domain-specific augmentations for forms
