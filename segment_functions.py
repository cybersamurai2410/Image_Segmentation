from transformers import pipeline, SamModel, SamProcessor
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Image Segmentation Model
sam_model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
sam_processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")

def load_image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

def show_colored_mask(mask, combined_mask, color):
    """
    Add a single-colored mask to the combined mask.
    Args:
        mask (numpy.ndarray): Binary mask to overlay.
        combined_mask (numpy.ndarray): Combined RGBA mask.
        color (tuple): RGBA color for the mask.
    """
    if mask.ndim == 3:  # If mask has channels then take the first one
        mask = mask[0]
    mask = mask.squeeze() # Remove extra dimension

    mask_binary = (mask > 0).astype(np.uint8)  # Ensure the mask is binary

    # Apply the color to the mask
    for c in range(3):  # RGB channels
        combined_mask[:, :, c] = np.where(mask_binary > 0, color[c], combined_mask[:, :, c])
    combined_mask[:, :, 3] = np.where(mask_binary > 0, color[3], combined_mask[:, :, 3]) # Alpha channel (transperency)

def segment_image(input_image, input_points):
    """
    Perform image segmentation and overlay masks with a single solid color.
    Args:
        input_image (PIL.Image): The input image.
        input_points (list): List of points [[x, y], ...].
    Returns:
        PIL.Image: Image with masks applied in one solid red color.
    """
    # Convert input points to a 4D tensor
    input_points_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

    # Process input and run the SAM model
    inputs = sam_processor(input_image, input_points=input_points_tensor, return_tensors="pt")
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Post-process masks
    predicted_masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )

    # Define a solid red color with full opacity
    single_color = (255, 0, 0, 100)  

    # Prepare a combined RGBA mask
    image_size = input_image.size
    combined_mask = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    # Apply all masks using the single color
    for mask in predicted_masks[0]:
        mask = mask.numpy()
        show_colored_mask(mask, combined_mask, single_color)

    # Combine the mask with the original image
    input_image_rgba = input_image.convert("RGBA") # Red Green Blue Alpha 
    combined_image = Image.alpha_composite(input_image_rgba, Image.fromarray(combined_mask, "RGBA"))

    return combined_image 
