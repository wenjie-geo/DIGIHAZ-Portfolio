"""
Assignment 1: Opening a Digital Image and Processing It
Module 04 - Topic #1: Unmanned Systems & Image Processing Basics

Purpose
-------
This script demonstrates how a grayscale digital image can be loaded,
represented as a NumPy array, modified by changing pixel values, displayed,
and saved as output image files.

The workflow follows the exercise example in DRR_Slides_01:
1. Load standard imaging libraries.
2. Load a test image using skimage.data.camera().
3. Visualize the original image.
4. Modify image content by manipulating the NumPy array.
5. Save the processed outputs.

Author: Chen Wenjie
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Included because the lecture example imports PIL.Image
from skimage import data


def main() -> None:
    """Run the complete image opening and processing workflow."""

    # ---------------------------------------------------------------------
    # 1. Create an output folder
    # ---------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # 2. Load a standard grayscale test image
    # ---------------------------------------------------------------------
    # The image is loaded as a NumPy array. Each pixel contains an intensity
    # value. For this uint8 grayscale image, the valid range is 0 to 255.
    A = data.camera()

    print("Image loaded successfully.")
    print("Image type:", type(A))
    print("Image shape:", A.shape)
    print("Data type:", A.dtype)
    print("Minimum pixel value:", A.min())
    print("Maximum pixel value:", A.max())

    # ---------------------------------------------------------------------
    # 3. Display and save the original image
    # ---------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(A, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Grayscale Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "01_original_image.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------
    # 4. Process the image by modifying a pixel region
    # ---------------------------------------------------------------------
    # Make a copy first so that the original image remains unchanged.
    A2 = np.copy(A)

    # Set a square region to 0. In a grayscale image, 0 represents black.
    A2[300:400, 300:400] = 0

    plt.figure(figsize=(6, 6))
    plt.imshow(A2, cmap="gray", vmin=0, vmax=255)
    plt.title("Processed Image: Modified Pixel Region")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "02_black_square_modification.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------
    # 5. Extra processing: grayscale inversion
    # ---------------------------------------------------------------------
    # Inversion changes dark pixels into bright pixels and bright pixels into
    # dark pixels. For uint8 images, the maximum value is 255.
    A3 = 255 - A

    plt.figure(figsize=(6, 6))
    plt.imshow(A3, cmap="gray", vmin=0, vmax=255)
    plt.title("Processed Image: Grayscale Inversion")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "03_grayscale_inversion.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------
    # 6. Compare the original and processed images in one figure
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(A, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(A2, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Black Square Modification")
    axes[1].axis("off")

    axes[2].imshow(A3, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Grayscale Inversion")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "04_comparison_figure.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------------------------------------------
    # 7. Save the raw image arrays as PNG files
    # ---------------------------------------------------------------------
    plt.imsave(output_dir / "original_image_raw.png", A, cmap="gray", vmin=0, vmax=255)
    plt.imsave(output_dir / "black_square_image_raw.png", A2, cmap="gray", vmin=0, vmax=255)
    plt.imsave(output_dir / "inverted_image_raw.png", A3, cmap="gray", vmin=0, vmax=255)

    print("All outputs have been saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()
