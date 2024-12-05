from transformers import pipeline, SamModel, SamProcessor
from PIL import Image
import torch
import gradio as gr
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

# Gradio App
with gr.Blocks() as demo:
    gr.Markdown("## AI Image Processing App")
    
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

    with gr.Tab("Object Detection with TTS"):
        object_input_image = gr.Image(type="pil", label="Input Image")
        detected_objects_image = gr.Image(type="pil", label="Image with Object Detection")
        detected_objects_audio = gr.Audio(type="numpy", label="Narrated Summary")
        detect_button = gr.Button("Detect Objects")

        detect_button.click(
            fn=detect_objects,
            inputs=object_input_image,
            outputs=[detected_objects_image, detected_objects_audio],
        )

if __name__ == "__main__":
    demo.launch(share=True)
