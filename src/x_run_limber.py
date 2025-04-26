import argparse
import torch
from x_limber_lightning import LimberGPTJLightning, ImageInput


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LIMBER model")
    parser.add_argument("--config", default="configs/clip_linear.yml", 
                        help="Path to config file")
    parser.add_argument("--image", required=True, 
                        help="Path or URL to input image")
    parser.add_argument("--prompt", default="A picture of", 
                        help="Text prompt to prepend")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum generation steps")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading model with config: {args.config}")
    # Initialize the Lightning model
    model = LimberGPTJLightning(args.config)
    
    # Move to GPU and use half precision for inference
    if torch.cuda.is_available():
        print("Using GPU for inference")
        model = model.cuda().half()
    else:
        print("WARNING: CUDA not available. Using CPU (this will be very slow)")
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Processing image: {args.image}")
    # Load and process the image
    img_input = ImageInput(args.image)
    
    print(f"Generating caption with prompt: '{args.prompt}'")
    # Preprocess the inputs and generate the caption
    inputs = model.preprocess_inputs([img_input, args.prompt])
    output = model.generate(
        embeddings=inputs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.max_steps
    )
    
    # Print the generated caption
    print("\nGenerated caption:")
    print(f"{args.prompt}{output[0]}")


if __name__ == "__main__":
    main()