import gradio as gr
from segment_functions import segment_image

def load_image_from_url(url):
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return image
    except Exception as e:
        return f"Error: {str(e)}"

selected_points = []  # Global list to store points

def capture_points(image, evt: gr.SelectData):
    """
    Capture click coordinates on the image.
    """
    global selected_points
    x, y = evt.index[0], evt.index[1]  # Extract x, y from Gradio click event
    selected_points.append([x, y])     # Append as [x, y] to list
    return str(selected_points)        # Display points as a string

def segment_image_ui(image):
    """
    Run the segmentation function using selected points.
    """
    global selected_points
    if not selected_points:
        return "Error: No points selected!"
    
    # Call your existing segment_image function
    segmented_image = segment_image(image, selected_points)
    selected_points = []  # Clear points after use
    return segmented_image

with gr.Blocks() as demo:
    gr.Markdown("# Image Segmentation")

    with gr.Row():
        # Image upload and URL input
        with gr.Column():
            image_input = gr.Image(sources=["upload"], type="pil", label="Upload Image")  
            image_url = gr.Textbox(label="Paste Image URL Here")
            load_button = gr.Button("Load Image from URL")
        
        image_output = gr.Image(type="pil", label="Segmented Image")

    # Selected points
    points_output = gr.Textbox(label="Selected Points (x, y)", interactive=False)

    # Button to run segmentation
    segment_button = gr.Button("Run Segmentation")

    # Load image from URL
    load_button.click(fn=load_image_from_url, inputs=[image_url], outputs=[image_input])

    # Capture click points
    image_input.select(fn=capture_points, inputs=image_input, outputs=points_output)

    # Run segmentation
    segment_button.click(fn=segment_image_ui, inputs=image_input, outputs=image_output)

demo.launch()
