"""
TUI Chat interface for testing the trained model.
Usage: uv run chat.py [--checkpoint PATH]
"""

import os
import sys
import argparse
import torch

# Verify macOS environment
def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon.")
    print("Environment verified: macOS with MPS available.")

verify_macos_env()

from prepare import Tokenizer, MAX_SEQ_LEN
from train import GPT, build_model_config


def load_checkpoint(checkpoint_path, device):
    """Load model and optimizer state from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def find_latest_checkpoint(checkpoints_dir="checkpoints"):
    """Find the most recent checkpoint in the directory."""
    if not os.path.exists(checkpoints_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
    return os.path.join(checkpoints_dir, checkpoints[-1])


def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate a response from the model."""
    device = next(model.parameters()).device
    
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(idx)
            logits = logits[:, -1, :] / temperature
            
            # Top-p (nucleus) sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            
            # Keep tokens with cumulative probability below top_p
            sorted_indices_to_remove = cumsum > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[0, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_token], dim=1)
            
            # Stop if we hit EOS (or a reserved token)
            if next_token.item() >= tokenizer.enc.n_vocab - 4:
                break
    
    # Decode the response (skip the prompt)
    response_tokens = idx[0][len(tokens):].tolist()
    return tokenizer.decode(response_tokens)


def chat_loop(model, tokenizer, args):
    """Main chat loop with simple TUI."""
    print("\n" + "="*50)
    print("  AutoResearch Chat Interface")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type 'reset' to clear conversation")
    print(f"  Generation: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}")
    print("="*50 + "\n")
    
    conversation = []
    
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            conversation = []
            print("[Conversation reset]")
            continue
        
        # Add to conversation
        conversation.append(f"User: {user_input}")
        
        # Build context from conversation (last few turns)
        context = "\n".join(conversation[-4:])  # Last 4 messages
        prompt = context + "\nAssistant:"
        
        # Generate response
        print("Assistant> ", end="", flush=True)
        response = generate_response(
            model, tokenizer, prompt, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(response)
        
        conversation.append(f"Assistant: {response}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained model")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint file (default: latest in checkpoints/)")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoint found!")
            print("Run training first to create checkpoints, or specify --checkpoint PATH")
            sys.exit(1)
    
    # Detect device
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    config_dict = checkpoint['config']
    
    # Build model config
    config = GPTConfig(
        sequence_len=config_dict['sequence_len'],
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_kv_head=config_dict['n_kv_head'],
        n_embd=config_dict['n_embd'],
        window_pattern=config_dict['window_pattern'],
    )
    
    # Create model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {config.n_layer} layers, {config.n_embd} embd dim")
    print(f"Trained for {checkpoint.get('total_training_time', 0)/3600:.1f} hours")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_directory()
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Start chat
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
