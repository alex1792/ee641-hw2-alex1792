"""
Script to generate and visualize samples from trained GAN generator.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the provided directory to path for imports
sys.path.append('../provided')

from models import Generator, Discriminator
from visualize import plot_alphabet_grid

def load_trained_generator(checkpoint_path, device, z_dim=100):
    """
    Load a trained generator from checkpoint.
    """
    # Create generator instance
    generator = Generator(z_dim=z_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Loaded generator from {checkpoint_path}")
    print(f"Training completed at epoch: {checkpoint['final_epoch']}")
    
    return generator

def generate_random_samples(generator, device, num_samples=16, seed=42):
    """
    Generate random samples from the generator.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    generator.eval()
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = [axes]
    elif grid_size > 1:
        axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random noise
            z = torch.randn(1, generator.z_dim).to(device)
            
            # Generate image - 檢查是否需要labels
            if hasattr(generator, 'conditional') and generator.conditional:
                label = torch.zeros(1, 26).to(device)
                fake_img = generator(z, label).squeeze().cpu()
            else:
                fake_img = generator(z).squeeze().cpu()
            
            # Convert from [-1, 1] to [0, 1] for display
            fake_img = (fake_img + 1) / 2
            
            # Plot
            if i < len(axes):
                axes[i].imshow(fake_img, cmap='gray', vmin=0, vmax=1)
                axes[i].set_title(f'Sample {i+1}', fontsize=10)
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Generated Font Samples (Random)', fontsize=16)
    plt.tight_layout()
    return fig

def main():
    """
    Main function to generate and visualize samples.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint_path = 'results/best_generator.pth'
    output_dir = 'results/generated_samples'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please run training first with train.py")
        return
    
    # Load trained generator
    generator = load_trained_generator(checkpoint_path, device)
    
    # Generate alphabet grid
    print("Generating alphabet grid...")
    fig1 = plot_alphabet_grid(generator, device=device, z_dim=generator.z_dim, seed=42)
    fig1.savefig(f'{output_dir}/generated_alphabet.png', dpi=300, bbox_inches='tight')
    print(f"Alphabet grid saved to {output_dir}/generated_alphabet.png")
    
    # Generate random samples
    print("Generating random samples...")
    fig2 = generate_random_samples(generator, device, num_samples=16, seed=42)
    fig2.savefig(f'{output_dir}/random_samples.png', dpi=300, bbox_inches='tight')
    print(f"Random samples saved to {output_dir}/random_samples.png")
    
    plt.show()
if __name__ == '__main__':
    main()