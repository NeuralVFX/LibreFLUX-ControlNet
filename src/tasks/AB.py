
import os
from PIL import Image
from functools import partial
from torchvision import transforms
from torch import nn
from ..util_pil import clean_image
from ..util import ensure_spatial_dims_div

def load_control_image_from_path(img_path: str) -> Image.Image:
    """
    Loads the control image from an adjacent directory by replacing 'img' with 'seg' in the path.
    """
    dir_path, filename = os.path.split(img_path)
    parent_dir, current_folder_name = os.path.split(dir_path)

    # Per your instruction, we use the direct replacement
    control_folder_name = current_folder_name.replace('img', 'seg')
    control_image_path = os.path.join(parent_dir, control_folder_name, filename)

    if not os.path.exists(control_image_path):
        raise FileNotFoundError(f"Control image not found at {control_image_path}")

    control_image = Image.open(control_image_path).convert("RGB")
    return control_image

def ab_base_transform(examples: dict, resolution: int) -> dict:
    """
    A custom transform function for the A/B task. It loads the control image
    directly and derives the main image path from the control path as a workaround
    for caching issues.
    """
    # Define standard transformations
    resize_smallest_edge = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    center_crop = transforms.CenterCrop(resolution)
    to_tensor = transforms.ToTensor()
    to_center = transforms.Normalize([0.5], [0.5])
    ensure_div_16 = partial(ensure_spatial_dims_div, n=16)

    def get_crop_params(img, tgt_size):
        y1 = max(0, int(round((img.height - tgt_size) / 2.0)))
        x1 = max(0, int(round((img.width - tgt_size) / 2.0)))
        return y1, x1

    # --- Use the control_img_path as the source of truth ---
    control_paths = examples["control_img_path"]
    if None in control_paths:
        raise ValueError("The 'control_img_path' column contains None values. Please check your data generation script.")

    # --- Workaround: Derive the main image paths from the control paths ---
    def get_main_path_from_control(control_path: str) -> str:
        dir_path, filename = os.path.split(control_path)
        parent_dir, control_folder_name = os.path.split(dir_path)
        main_folder_name = control_folder_name.replace('seg', 'img', 1)
        return os.path.join(parent_dir, main_folder_name, filename)

    main_image_paths = [get_main_path_from_control(path) for path in control_paths]

    # Load both sets of images
    images = [Image.open(path).convert("RGB") for path in main_image_paths]
    cond_images = [Image.open(path).convert("RGB") for path in control_paths]

    # --- ⭐ FIX: Apply the SAME initial processing to BOTH image lists ---
    images = [clean_image(img) for img in images]
    cond_images = [clean_image(img) for img in cond_images] # Apply clean_image to cond_images

    # ----- identical spatial pipeline -------------------------------------
    images      = [center_crop(resize_smallest_edge(img)) for img in images]
    cond_images = [center_crop(resize_smallest_edge(img)) for img in cond_images]

    # metadata
    img_params  = [(img.height, img.width) for img in images]
    crop_params = [get_crop_params(img, resolution) for img in images]


    # --- Convert both to tensors ---
    images_torch = [to_center(ensure_div_16(to_tensor(img))) for img in images]
    cond_images_torch = [to_center(ensure_div_16(to_tensor(img))) for img in cond_images]

    # Populate the final dictionary
    examples["pixel_values"] = images_torch
    examples["conditioning_pixel_values"] = cond_images_torch
    examples["img_params"] = img_params
    examples["crop_params"] = crop_params

    return examples

def add_transform_to_dataset(dataset, resolution, accelerator):
    """Adds the A/B transform to the dataset."""
    transform_fn = partial(ab_base_transform, resolution=resolution)

    if accelerator is not None:
        with accelerator.main_process_first():
            dataset = dataset.with_transform(transform_fn)
    else:
        dataset = dataset.with_transform(transform_fn)

    return dataset


