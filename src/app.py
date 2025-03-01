import os
import gradio as gr
from extractor import TextExtractor

# Initialize the extractor
extractor = TextExtractor()
os.makedirs("./uploads", exist_ok=True)

def process_image(image, system_prompt, user_prompt):
    """Process the uploaded image and extract text"""
    # Save the uploaded image temporarily
    temp_path = os.path.join("./uploads", "temp_image.png")
    image.save(temp_path)
    
    # Use custom prompts if provided, otherwise use defaults
    custom_system = system_prompt if system_prompt.strip() else None
    custom_user = user_prompt if user_prompt.strip() else None
    
    # Extract text from the image
    extracted_text = extractor.extract_text(
        temp_path, 
        system_prompt=custom_system,
        user_prompt=custom_user
    )
    
    return extracted_text

# Create the Gradio interface
with gr.Blocks(title="Image Text Extractor") as demo:
    gr.Markdown("# Image Text Extractor\nUpload an image to extract text using Qwen2.5-VL")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            
            with gr.Accordion("Advanced Options", open=False):
                system_prompt = gr.Textbox(
                    label="System Prompt (optional)", 
                    placeholder="Leave empty to use default system prompt",
                    lines=2
                )
                user_prompt = gr.Textbox(
                    label="User Prompt (optional)", 
                    placeholder="Leave empty to use default user prompt",
                    lines=2
                )
            
            extract_button = gr.Button("Extract Text")
        
        with gr.Column():
            output_text = gr.Textbox(label="Extracted Text", lines=20)
    
    extract_button.click(
        fn=process_image,
        inputs=[input_image, system_prompt, user_prompt],
        outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    # Start the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=8080)