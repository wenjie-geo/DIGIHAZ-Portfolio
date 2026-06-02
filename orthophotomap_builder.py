"""
Assignment 4: Orthophotomap Generation from DSM and UAV Imagery
Module 04 - Topic #4: Drone-Based Damage Assessment Applications

This script follows the exercise workflow in the lecture slides:

1. Load the Digital Elevation Model / Digital Surface Model.
2. Load the UAV imagery data.
3. Use the collinearity condition to map X,Y,Z from the DSM to the image data.
4. Display and save the generated orthophotomap.

Expected input files in the same folder:
- DJI_0106.jpg
- dsm_10cm.tif
- IO_EO.xls

Author: Chen Wenjie
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from PIL import Image


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
WORK_DIR = Path(".")
UAV_IMAGE_FILE = WORK_DIR / "DJI_0106.jpg"
DSM_FILE = WORK_DIR / "dsm_10cm.tif"
IO_EO_FILE = WORK_DIR / "IO_EO.xls"
OUTPUT_DIR = WORK_DIR / "outputs"

# Downsampling factor for DSM.
# step=5 means using every 5th DSM pixel, which converts 10 cm DSM to ~50 cm
# for faster classroom execution. Use step=1 only if the computer is fast enough.
DSM_STEP = 5


def check_input_files() -> None:
    """Check whether required files exist."""
    required_files = [UAV_IMAGE_FILE, DSM_FILE, IO_EO_FILE]
    print("Working directory:", WORK_DIR.resolve())

    for file_path in required_files:
        print(f"{file_path.name}: exists={file_path.exists()}")
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {file_path.name}. Put it in the same folder as this script."
            )


def load_orientation_table() -> pd.DataFrame:
    """Load and display the IO/EO spreadsheet."""
    orientation_table = pd.read_excel(IO_EO_FILE, header=None, engine="xlrd")
    print("IO_EO.xls loaded successfully.")
    print("Shape:", orientation_table.shape)
    print(orientation_table)
    return orientation_table


def load_uav_image() -> np.ndarray:
    """Load the UAV image as an RGB NumPy array."""
    image = Image.open(UAV_IMAGE_FILE).convert("RGB")
    uav_image = np.array(image)

    print("UAV image loaded successfully.")
    print("UAV image shape:", uav_image.shape)

    return uav_image


def load_dsm():
    """Load the DSM GeoTIFF and return the array and georeferencing metadata."""
    with rasterio.open(DSM_FILE) as src:
        dsm_full = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata

    print("DSM loaded successfully.")
    print("DSM shape:", dsm_full.shape)
    print("CRS:", crs)
    print("Bounds:", bounds)
    print("NoData:", nodata)
    print("Minimum elevation:", np.nanmin(dsm_full))
    print("Maximum elevation:", np.nanmax(dsm_full))

    return dsm_full, transform, crs, bounds, nodata


def rotation_matrix(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    """
    Construct a rotation matrix from omega, phi, and kappa.

    The order follows the lecture slide:
    R = R_kappa @ R_phi @ R_omega
    """
    omega = np.deg2rad(omega_deg)
    phi = np.deg2rad(phi_deg)
    kappa = np.deg2rad(kappa_deg)

    r_omega = np.array([
        [1, 0, 0],
        [0, np.cos(omega), np.sin(omega)],
        [0, -np.sin(omega), np.cos(omega)]
    ])

    r_phi = np.array([
        [np.cos(phi), 0, -np.sin(phi)],
        [0, 1, 0],
        [np.sin(phi), 0, np.cos(phi)]
    ])

    r_kappa = np.array([
        [np.cos(kappa), np.sin(kappa), 0],
        [-np.sin(kappa), np.cos(kappa), 0],
        [0, 0, 1]
    ])

    return r_kappa @ r_phi @ r_omega


def prepare_ground_coordinates(dsm_full: np.ndarray, transform, step: int):
    """
    Downsample the DSM and prepare X, Y, Z coordinate arrays.

    X and Y are derived from the DSM affine transform.
    Z is the DSM elevation value.
    """
    dsm = dsm_full[::step, ::step]
    rows, cols = dsm.shape

    row_indices = np.arange(0, dsm_full.shape[0], step)
    col_indices = np.arange(0, dsm_full.shape[1], step)

    row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing="ij")

    # Convert raster row/column to projected coordinates using affine transform.
    x_ground = (
        transform.c
        + (col_grid + 0.5) * transform.a
        + (row_grid + 0.5) * transform.b
    )
    y_ground = (
        transform.f
        + (col_grid + 0.5) * transform.d
        + (row_grid + 0.5) * transform.e
    )
    z_ground = dsm

    print("Ground coordinate arrays prepared.")
    print("X shape:", x_ground.shape)
    print("Y shape:", y_ground.shape)
    print("Z shape:", z_ground.shape)
    print("X range:", np.nanmin(x_ground), np.nanmax(x_ground))
    print("Y range:", np.nanmin(y_ground), np.nanmax(y_ground))
    print("Z range:", np.nanmin(z_ground), np.nanmax(z_ground))

    return dsm, x_ground, y_ground, z_ground


def project_dem_to_image(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    X0: float,
    Y0: float,
    Z0: float,
    R: np.ndarray,
    c_mm: float,
    x0_mm: float,
    y0_mm: float,
    pixel_size_mm: float,
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
):
    """
    Project DSM coordinates X,Y,Z to UAV image pixel coordinates
    using the collinearity condition.
    """
    dX = X - X0
    dY = Y - Y0
    dZ = Z - Z0

    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]

    denominator = r31 * dX + r32 * dY + r33 * dZ
    denominator = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)

    # Photo coordinates in millimetres.
    x_photo_mm = x0_mm - c_mm * (
        (r11 * dX + r12 * dY + r13 * dZ) / denominator
    )
    y_photo_mm = y0_mm - c_mm * (
        (r21 * dX + r22 * dY + r23 * dZ) / denominator
    )

    # Convert photo coordinates to original full-resolution image pixels.
    col_original = (original_width / 2) + (x_photo_mm / pixel_size_mm)
    row_original = (original_height / 2) - (y_photo_mm / pixel_size_mm)

    # Convert to the actual image size used in this exercise.
    col_actual = col_original * scale_x
    row_actual = row_original * scale_y

    print("Projection completed.")
    print("Projected column range:", np.nanmin(col_actual), np.nanmax(col_actual))
    print("Projected row range:", np.nanmin(row_actual), np.nanmax(row_actual))

    return col_actual, row_actual


def build_orthophoto(uav_image: np.ndarray, col_img: np.ndarray, row_img: np.ndarray):
    """Sample UAV image pixels and generate an orthophotomap array."""
    rows, cols = col_img.shape
    actual_height, actual_width, _ = uav_image.shape

    orthophoto = np.zeros((rows, cols, 3), dtype=np.uint8)

    finite = np.isfinite(col_img) & np.isfinite(row_img)

    col_px = np.zeros_like(col_img, dtype=np.int64)
    row_px = np.zeros_like(row_img, dtype=np.int64)

    col_px[finite] = np.rint(col_img[finite]).astype(np.int64)
    row_px[finite] = np.rint(row_img[finite]).astype(np.int64)

    valid = (
        finite
        & (col_px >= 0)
        & (col_px < actual_width)
        & (row_px >= 0)
        & (row_px < actual_height)
    )

    orthophoto[valid] = uav_image[row_px[valid], col_px[valid]]

    print("Orthophotomap generated.")
    print("Valid projected pixels:", int(np.sum(valid)))
    print("Total pixels:", rows * cols)
    print("Valid percentage:", float(np.sum(valid) / (rows * cols) * 100), "%")

    return orthophoto, valid


def crop_valid_area(orthophoto: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Crop the orthophotomap to the valid projected area."""
    valid_rows, valid_cols = np.where(valid)

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        print("No valid projected pixels found. Returning original orthophoto.")
        return orthophoto

    r_min, r_max = valid_rows.min(), valid_rows.max()
    c_min, c_max = valid_cols.min(), valid_cols.max()

    cropped = orthophoto[r_min:r_max + 1, c_min:c_max + 1]
    print("Cropped orthophotomap shape:", cropped.shape)

    return cropped


def save_and_display_results(dsm, uav_image, orthophoto, cropped) -> None:
    """Save output images and display summary figures."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    Image.fromarray(orthophoto).save(OUTPUT_DIR / "orthophotomap_result.png")
    Image.fromarray(cropped).save(OUTPUT_DIR / "orthophotomap_result_cropped.png")

    plt.figure(figsize=(10, 8))
    plt.imshow(orthophoto)
    plt.title("Generated Orthophotomap from DSM and UAV Image")
    plt.axis("off")
    plt.savefig(OUTPUT_DIR / "orthophotomap_result_figure.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(cropped)
    plt.title("Cropped Orthophotomap from DSM and UAV Image")
    plt.axis("off")
    plt.savefig(OUTPUT_DIR / "orthophotomap_result_cropped_figure.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(dsm, cmap="terrain")
    axes[0].set_title("Loaded DSM")
    axes[0].axis("off")

    axes[1].imshow(uav_image)
    axes[1].set_title("Loaded UAV Image")
    axes[1].axis("off")

    axes[2].imshow(cropped)
    axes[2].set_title("Generated Orthophotomap")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_result.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved outputs:")
    print(OUTPUT_DIR / "orthophotomap_result.png")
    print(OUTPUT_DIR / "orthophotomap_result_cropped.png")
    print(OUTPUT_DIR / "comparison_result.png")


def main() -> None:
    """Run the full orthophotomap workflow."""
    check_input_files()
    load_orientation_table()

    uav_image = load_uav_image()
    dsm_full, transform, crs, bounds, nodata = load_dsm()

    # ------------------------------------------------------------------
    # Camera parameters from IO_EO.xls / lecture slide
    # ------------------------------------------------------------------
    original_width = 8192
    original_height = 5460

    actual_height, actual_width, _ = uav_image.shape

    scale_x = actual_width / original_width
    scale_y = actual_height / original_height

    print("Actual image width:", actual_width)
    print("Actual image height:", actual_height)
    print("Scale X:", scale_x)
    print("Scale Y:", scale_y)

    # Interior orientation, in millimetres.
    c_mm = 35.9235
    x0_mm = -0.1130
    y0_mm = -0.0680
    pixel_size_mm = 0.0044

    # Exterior orientation.
    X0 = 302412.9194
    Y0 = 4134628.6581
    Z0 = 655.9317

    omega_deg = 0.2158
    phi_deg = 0.2202
    kappa_deg = -12.62469

    R = rotation_matrix(omega_deg, phi_deg, kappa_deg)
    print("Rotation matrix:")
    print(R)

    dsm, X_ground, Y_ground, Z_ground = prepare_ground_coordinates(
        dsm_full=dsm_full,
        transform=transform,
        step=DSM_STEP
    )

    col_img, row_img = project_dem_to_image(
        X=X_ground,
        Y=Y_ground,
        Z=Z_ground,
        X0=X0,
        Y0=Y0,
        Z0=Z0,
        R=R,
        c_mm=c_mm,
        x0_mm=x0_mm,
        y0_mm=y0_mm,
        pixel_size_mm=pixel_size_mm,
        scale_x=scale_x,
        scale_y=scale_y,
        original_width=original_width,
        original_height=original_height,
    )

    orthophoto, valid = build_orthophoto(uav_image, col_img, row_img)
    cropped = crop_valid_area(orthophoto, valid)

    save_and_display_results(dsm, uav_image, orthophoto, cropped)


if __name__ == "__main__":
    main()
