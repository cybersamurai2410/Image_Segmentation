import gradio as gr
from segment_functions import segment_image, estimate_depth, detect_objects

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

demo.launch()
