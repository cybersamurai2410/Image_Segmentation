from transformers import pipeline, SamModel, SamProcessor
from PIL import Image
import torch
import numpy as np
from helper import show_mask_on_image, render_results_in_image, summarize_predictions_natural_language

# Image Segmentation Model
sam_model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
sam_processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")

# Depth Estimation Model
depth_estimator = pipeline(task="depth-estimation", model="./models/Intel/dpt-hybrid-midas")

# Object Detection Model
od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

# Text-to-Speech Model
tts_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")

def segment_image(input_image, input_points):
    """
    Perform image segmentation using SAM.
    """
    inputs = sam_processor(input_image, input_points=input_points, return_tensors="pt")
    with torch.no_grad():
        outputs = sam_model(**inputs)
    predicted_masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )
    for i in range(predicted_masks.shape[1]):  
        mask = predicted_masks[0, i].numpy()
        input_image = show_mask_on_image(input_image, mask)
    return input_image

def estimate_depth(input_image):
    """
    Perform depth estimation on the input image.
    """
    output = depth_estimator(input_image)
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

def detect_objects(input_image):
    """
    Perform object detection and generate TTS summary.
    """
    pipeline_output = od_pipe(input_image)
    processed_image = render_results_in_image(input_image, pipeline_output)
    summary = summarize_predictions_natural_language(pipeline_output)
    narrated_text = tts_pipe(summary)["audio"][0]
    return processed_image, narrated_text
