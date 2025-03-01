import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# Configure PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class TextExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="./models"):
        """
        Initialize the Text Extractor with the specified model.
        
        Args:
            model_name (str): HuggingFace model name
            cache_dir (str): Directory to cache the downloaded model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Default prompts
        self.default_system_prompt = (
            "You are an expert in extracting text from images. "
            "Your task is to transcribe all text from the provided image exactly as it appears. "
        )
        
        self.default_user_prompt = (
            "Please transcribe all text from this image exactly as it appears. "
            "Preserve the layout, formatting, and structure of tables, lists, and paragraphs. "
            "Include all visible text, whether printed or handwritten."
        )
    
    def extract_text(self, image_path, system_prompt=None, user_prompt=None, max_tokens=1024):
        """
        Extract text from an image file.
        
        Args:
            image_path (str): Path to the local image file
            system_prompt (str, optional): Custom system prompt
            user_prompt (str, optional): Custom user prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Extracted text from the image
        """
        # Use default prompts if not provided
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        
        if user_prompt is None:
            user_prompt = self.default_user_prompt
        
        # Load image from file path
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return f"Error loading image: {str(e)}"
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        # Process the vision information
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=False,
        )
        
        # Decode the generated IDs to text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def batch_extract(self, image_paths, system_prompt=None, user_prompt=None, max_tokens=1024):
        """
        Extract text from multiple images.
        
        Args:
            image_paths (list): List of image file paths
            system_prompt (str, optional): Custom system prompt
            user_prompt (str, optional): Custom user prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            dict: Dictionary mapping image paths to extracted text
        """
        results = {}
        for path in image_paths:
            results[path] = self.extract_text(path, system_prompt, user_prompt, max_tokens)
        return results


if __name__ == "__main__":
    # Example usage
    extractor = TextExtractor()
    
    # Single image extraction
    image_path = "./data/samples/example.jpg"  # Change this to your image path
    extracted_text = extractor.extract_text(image_path)
    
    print(f"Extracted Text from {image_path}:")
    print(extracted_text)