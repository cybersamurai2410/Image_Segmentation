from transformers import pipeline, SamModel, SamProcessor
import torch
from PIL import Image
import gradio as gr
import numpy as np
from helper import show_mask_on_image

# Initialize Models
# SAM (Segment Anything Model) for image segmentation
sam_model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
sam_processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")

# Depth Estimator
depth_estimator = pipeline(task="depth-estimation", model="./models/Intel/dpt-hybrid-midas")

# Helper Functions
def segment_image(input_image, input_points):
    """
    Perform image segmentation using SAM.
    Args:
        input_image (PIL.Image): Input image.
        input_points (list of lists): Points to guide segmentation.
    Returns:
        PIL.Image: Segmented image with masks overlaid.
    """
    # Preprocess image and points
    inputs = sam_processor(input_image, input_points=input_points, return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Post-process masks
    predicted_masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )

    # Overlay masks on the original image
    for i in range(predicted_masks.shape[1]):  # Iterate over masks
        mask = predicted_masks[0, i].numpy()
        input_image = show_mask_on_image(input_image, mask)

    return input_image

def estimate_depth(input_image):
    """
    Perform depth estimation on the input image.
    Args:
        input_image (PIL.Image): Input image.
    Returns:
        PIL.Image: Depth map.
    """
    # Estimate depth
    output = depth_estimator(input_image)

    # Rescale and format depth map
    prediction = torch.nn.functional.interpolate(
        output["predicted_depth"].unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_map = Image.fromarray(formatted)
  
    return depth_map

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Segmentation and Depth Estimation")

    with gr.Tab("Image Segmentation"):
        input_image = gr.Image(type="pil", label="Input Image")
        input_points = gr.Textbox(label="Input Points (e.g., [[160, 70]])", value="[[160, 70]]")
        segmented_image = gr.Image(type="pil", label="Segmented Image")
        segment_button = gr.Button("Segment Image")

        segment_button.click(
            fn=lambda img, pts: segment_image(img, eval(pts)),
            inputs=[input_image, input_points],
            outputs=segmented_image,
        )

    with gr.Tab("Depth Estimation"):
        depth_input_image = gr.Image(type="pil", label="Input Image")
        depth_output_image = gr.Image(type="pil", label="Depth Map")
        depth_button = gr.Button("Estimate Depth")

        depth_button.click(
            fn=estimate_depth,
            inputs=depth_input_image,
            outputs=depth_output_image,
        )

if __name__ == "__main__":
    demo.launch(share=True)
