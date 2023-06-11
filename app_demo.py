import gradio as gr
from PIL import Image

from my_pipeline import *



def process_image(input_image, output_path = "fale.png"):
    # Perform image processing on the input image
    # Replace this with your own image processing code
    img = Image.open(input_image).convert('RGB')
    original_size = img.size
    transform_params = get_params(opt, original_size)
    transforms = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1) )
    img = transforms(img) # Example: convert the image to grayscale
    img = torch.unsqueeze(img, 0)
    fake = model.netG(img.to("cpu"))
    _fake = tensor2im(fake)
    save_image(_fake, output_path)
    return output_path

# Create the input and output interfaces as blocks

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            input_image = gr.Image(label="Input Image", type="filepath").style(height=600, width=700)
            
        with gr.Column(scale=1, min_width=600):
            output_image = gr.Image(label="Output Image", type="filepath").style(height=600, width=700)

    btn = gr.Button("Colorize")
    btn.click(process_image, inputs=input_image, outputs=output_image)  

# Launch the Gradio interface
demo.launch(enable_queue=True, server_port=1402, server_name="0.0.0.0", share=True)