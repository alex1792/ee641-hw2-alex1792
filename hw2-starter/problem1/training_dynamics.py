"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

sys.path.append('../provided')
from metrics import mode_coverage_score
from visualize import plot_training_history, plot_alphabet_grid

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda', d_learning_rate=0.0002, g_learning_rate=0.0002):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)

    # set models to train mode
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            d_optimizer.zero_grad()
            # 2. Forward pass on real images
            real_output = discriminator(real_images)
            # 3. Compute real loss
            d_real_loss = criterion(real_output, real_labels)
            # 4. Generate fake images from random z
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z, labels)
            # 5. Forward pass on fake images (detached)
            fake_output = discriminator(fake_images.detach())
            # 6. Compute fake loss
            d_fake_loss = criterion(fake_output, fake_labels)
            # 7. Backward and optimize
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            # 8. Compute gradient norm
            d_grad_norm = 0.0
            for param in discriminator.parameters():
                if param.grad is not None:
                    d_grad_norm += param.grad.data.norm(2).item() ** 2
            d_grad_norm = d_grad_norm ** 0.5
            d_optimizer.step()
            
            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            g_optimizer.zero_grad()
            # 2. Generate fake images
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z)
            # 3. Forward pass through discriminator
            fake_output = discriminator(fake_images)
            # 4. Compute adversarial loss
            g_loss = criterion(fake_output, real_labels)
            # 5. Backward and optimize
            g_loss.backward()
            # 6. Compute gradient norm
            g_grad_norm = 0.0
            for param in generator.parameters():
                if param.grad is not None:
                    g_grad_norm += param.grad.data.norm(2).item() ** 2
            g_grad_norm = g_grad_norm ** 0.5
            g_optimizer.step()

            # 7. update epoch losses
            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
            num_batches += 1
            
            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['d_grad_norm'].append(d_grad_norm)
                history['g_grad_norm'].append(g_grad_norm)
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # print epoch losses
        avg_d_loss = d_epoch_loss / num_batches
        avg_g_loss = g_epoch_loss / num_batches
        print(f"Epoch {epoch + 1}: D Loss = {avg_d_loss:.4f}, G Loss = {avg_g_loss:.4f}")


        # Analyze mode collapse every 10 epochs
        if (epoch + 1) % 10 == 0:
            mode_coverage, letter_analysis = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append(mode_coverage)
            history['mode_coverage_epoch'].append(epoch)
            history['letter_analysis'].append(letter_analysis)
            print(f"Epoch {epoch + 1}: Mode coverage = {mode_coverage:.2f}")
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
        dict: letter analysis
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)
    
    generator.eval()

    with torch.no_grad():
        # generate n_samples images
        z = torch.randn(n_samples, generator.z_dim).to(device)
        generated_images = generator(z)

        coverage_results = mode_coverage_score(generated_images)

        coverage_score = coverage_results['coverage_score']
        letter_counts = coverage_results['letter_counts']
        missing_letters = coverage_results['missing_letters']
        n_unique = coverage_results['n_unique']

        # Create letter analysis
        letter_analysis = {}
        for i in range(26):
            letter = chr(65 + i)  # A-Z
            if i in letter_counts:
                letter_analysis[letter] = {
                    'count': letter_counts[i],
                    'percentage': letter_counts[i] / n_samples * 100,
                    'status': 'survived'
                }
            else:
                letter_analysis[letter] = {
                    'count': 0,
                    'percentage': 0.0,
                    'status': 'disappeared'
                }

        print(f"Mode coverage score: {coverage_score:.2f}")
        print(f"Letter counts: {letter_counts}")
        print(f"Missing letters: {missing_letters}")
        print(f"Number of unique letters: {n_unique}")
        
    generator.train()
    return coverage_score, letter_analysis

def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear
    fig = plot_training_history(history, save_path)

    # letter analysis is preserved in the history
    if 'letter_analysis' in history:
        survival_plot_path = save_path.replace('.png', '_survival_plot.png')
        plot_survival_plot(history, survival_plot_path)
    return fig

def plot_survival_plot(history, save_path):
    epochs = history['mode_coverage_epoch']
    letter_analyses = history['letter_analysis']
    
    # Create a matrix to track letter survival over time
    letters = [chr(65 + i) for i in range(26)]  # A-Z
    survival_matrix = np.zeros((len(letters), len(epochs)))
    
    # Fill the survival matrix
    for epoch_idx, letter_analysis in enumerate(letter_analyses):
        for letter_idx, letter in enumerate(letters):
            if letter in letter_analysis:
                survival_matrix[letter_idx, epoch_idx] = letter_analysis[letter]['count']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # ========== Subplot 1: Heatmap ==========
    im = ax1.imshow(survival_matrix, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Epoch (every 10 epochs)')
    ax1.set_ylabel('Letters (A-Z)')
    ax1.set_title('Letter Survival Over Training (Count of Generated Samples)', fontsize=14)
    ax1.set_yticks(range(26))
    ax1.set_yticklabels(letters)
    ax1.set_xticks(range(len(epochs)))
    ax1.set_xticklabels(epochs)
    
    # Add colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Number of Generated Samples')
    
    # ========== Subplot 2: Line Plot ==========
    # Plot survival rate for each letter
    for letter_idx, letter in enumerate(letters):
        letter_counts = survival_matrix[letter_idx, :]
        # Normalize by the maximum count for this letter across all epochs
        max_count = letter_counts.max()
        if max_count > 0:
            survival_rate = letter_counts / max_count
        else:
            survival_rate = np.zeros_like(letter_counts)
        
        # Color code: green for letters that survive, red for those that disappear
        if max_count > 0:
            color = 'green' if survival_rate[-1] > 0.1 else 'red'  # Last epoch survival rate
            alpha = 0.7 if survival_rate[-1] > 0.1 else 0.4
        else:
            color = 'red'
            alpha = 0.4
            
        ax2.plot(epochs, survival_rate, label=letter, alpha=alpha, linewidth=1.5, color=color)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Normalized Survival Rate')
    ax2.set_title('Normalized Letter Survival Rate Over Training', fontsize=14)
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3)
    
    # Add legend (only show a few letters to avoid clutter)
    legend_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    handles = [plt.Line2D([0], [0], color='green', alpha=0.7, linewidth=1.5, label='Surviving Letters'),
               plt.Line2D([0], [0], color='red', alpha=0.4, linewidth=1.5, label='Disappeared Letters')]
    ax2.legend(handles=handles, loc='upper right')
    
    # Add text annotation for interpretation
    ax2.text(0.02, 0.98, 'Green lines: Letters that survive\nRed lines: Letters that disappear', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Letter survival analysis saved to {save_path}")