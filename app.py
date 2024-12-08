import gradio as gr
from segment_functions import segment_image, show_colored_mask, load_image_from_url

selected_points = []  # Global list to store points

def capture_points(image, evt: gr.SelectData):
    """
    Capture click coordinates on the image.
    """
    global selected_points
    x, y = evt.index[0], evt.index[1]  # Extract x, y from Gradio event
    selected_points.append([x, y])     # Append as [x, y] to list
    return str(selected_points)        # Display points as a string

def segment_image_ui(image):
    """
    Wrapper to call segment_image function.
    """
    global selected_points
    if not selected_points:
        return "Error: No points selected!"
    # Call the existing segment_image function
    segmented_image = segment_image(image, selected_points)
    selected_points = []  # Clear the points after segmentation
    return segmented_image

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Segmentation App")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image & Click Points")
        image_output = gr.Image(type="pil", label="Segmented Image")
    
    points_output = gr.Textbox(label="Selected Points (x, y)", interactive=False)
    segment_button = gr.Button("Run Segmentation")

    # Capture click points
    image_input.select(fn=capture_points, inputs=image_input, outputs=points_output)

    # Run segmentation
    segment_button.click(fn=segment_image_ui, inputs=image_input, outputs=image_output)

demo.launch()
