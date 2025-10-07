"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # KL annealing schedule
    def kl_anneal_schedule(epoch):
        """
        TODO: Implement KL annealing schedule
        Start with beta ≈ 0, gradually increase to 1.0
        Consider cyclical annealing for better results
        """
        cycle_length = num_epochs // 3
        cycle_position = epoch % cycle_length
        beta = min(1.0, 2 * cycle_position / cycle_length)
        return beta
    
    # Free bits threshold
    free_bits = 0.5  # Minimum nats per latent dimension
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        beta = kl_anneal_schedule(epoch)

        epoch_losses = []
        epoch_recon_losses = []
        epoch_kl_low_losses = []
        epoch_kl_high_losses = []
        
        for batch_idx, patterns in enumerate(data_loader):
            patterns = patterns.to(device)
            
            # TODO: Implement training step
            # 1. Forward pass through hierarchical VAE
            optimizer.zero_grad()
            pattern_logits, total_loss, mu_low, logvar_low, mu_high, logvar_high = model(patterns, beta=beta)

            # 2. Compute reconstruction loss
            recon_loss = nn.functional.binary_cross_entropy_with_logits(pattern_logits, patterns.float())

            # 3. Compute KL divergences (both levels)
            kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp(), dim=1)
            kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp(), dim=1)
            
            # 4. Apply free bits to prevent collapse
            kl_low_free_bits = torch.clamp(kl_low, min=free_bits * model.z_low_dim)
            kl_high_free_bits = torch.clamp(kl_high, min=free_bits * model.z_high_dim)

            # 5. Total loss = recon_loss + beta * kl_loss
            kl_loss = kl_low_free_bits.mean() + kl_high_free_bits.mean()
            total_loss = recon_loss + beta * kl_loss
            
            # 6. Backward and optimize
            total_loss.backward()
            optimizer.step()

            # preserve loss
            epoch_losses.append(total_loss.item())
            epoch_recon_losses.append(recon_loss.item())
            epoch_kl_low_losses.append(kl_low_free_bits.mean().item())
            epoch_kl_high_losses.append(kl_high_free_bits.mean().item())
        
        # preserve epoch stats
        history['epoch'].append(epoch)
        history['total_loss'].append(np.mean(epoch_losses))
        history['recon_loss'].append(np.mean(epoch_recon_losses))
        history['kl_low'].append(np.mean(epoch_kl_low_losses))
        history['kl_high'].append(np.mean(epoch_kl_high_losses))
        history['beta'].append(beta)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss={np.mean(epoch_losses):.4f}, "
                  f"Recon={np.mean(epoch_recon_losses):.4f}, "
                  f"KL_low={np.mean(epoch_kl_low_losses):.4f}, "
                  f"KL_high={np.mean(epoch_kl_high_losses):.4f}, "
                  f"Beta={beta:.3f}")
            
    
    return history

def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda'):
    """
    Generate diverse drum patterns using the hierarchy.
    
    TODO:
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
    model.eval()
    patterns = []

    with torch.no_grad():
        for style_idx in range(n_styles):
            style_patterns = []

            # 1. Sample n_styles from z_high prior
            z_high = torch.randn(1, model.z_high_dim).to(device) 

            for var_idx in range(n_variations):
                # 2. For each style, sample n_variations from conditional p(z_low|z_high)
                z_low = model.decoder_high(z_high)

                noise = torch.randn_like(z_low) * 0.1
                z_low += noise

                # 3. Decode to patterns
                pattern_logits = model.decode_hierarchy(z_high, z_low)
                pattern = torch.sigmoid(pattern_logits).cpu()
                style_patterns.append(pattern.squeeze(0))
            
            patterns.append(style_patterns)
    
    return patterns

def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Diagnose which latent dimensions are being used.
    
    TODO:
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    model.eval()
    
    all_kl_low = []
    all_kl_high = []
    
    with torch.no_grad():
        for patterns in data_loader:
            patterns = patterns.to(device)
            
            # 1. Encode validation data
            _, _, mu_high, logvar_high, mu_low, logvar_low = model.encode_hierarchy(patterns)
            
            # 2. Compute KL divergence per dimension
            # KL per dimension for low-level latent
            kl_low_per_dim = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
            all_kl_low.append(kl_low_per_dim)
            
            # KL per dimension for high-level latent
            kl_high_per_dim = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
            all_kl_high.append(kl_high_per_dim)
    
    # Concatenate all batches
    all_kl_low = torch.cat(all_kl_low, dim=0)
    all_kl_high = torch.cat(all_kl_high, dim=0)
    
    # 3. Identify collapsed dimensions (KL ≈ 0)
    # Compute mean KL per dimension
    mean_kl_low = all_kl_low.mean(dim=0)
    mean_kl_high = all_kl_high.mean(dim=0)
    
    # 4. Return utilization statistics
    collapse_threshold = 0.01  # Dimensions with KL < 0.01 are considered collapsed
    
    low_utilization = (mean_kl_low > collapse_threshold).float().mean().item()
    high_utilization = (mean_kl_high > collapse_threshold).float().mean().item()
    
    stats = {
        'low_level_utilization': low_utilization,
        'high_level_utilization': high_utilization,
        'mean_kl_low_per_dim': mean_kl_low.cpu().numpy(),
        'mean_kl_high_per_dim': mean_kl_high.cpu().numpy(),
        'collapsed_low_dims': (mean_kl_low < collapse_threshold).sum().item(),
        'collapsed_high_dims': (mean_kl_high < collapse_threshold).sum().item(),
        'total_low_dims': model.z_low_dim,
        'total_high_dims': model.z_high_dim
    }
    
    print(f"Low-level latent utilization: {low_utilization:.2%}")
    print(f"High-level latent utilization: {high_utilization:.2%}")
    print(f"Collapsed low-level dimensions: {stats['collapsed_low_dims']}/{model.z_low_dim}")
    print(f"Collapsed high-level dimensions: {stats['collapsed_high_dims']}/{model.z_high_dim}")
    
    return stats

def kl_annealing_schedule(epoch, method='cyclical', total_epochs=100):
    """
    KL annealing schedule to prevent posterior collapse.
    
    Args:
        epoch: Current epoch
        method: 'linear', 'cyclical', or 'sigmoid'
        total_epochs: Total number of epochs
        
    Returns:
        beta: KL weight for current epoch
    """
    if method == 'linear':
        # Linear increase from 0 to 1
        return min(1.0, epoch / (total_epochs * 0.5))
    
    elif method == 'cyclical':
        # Cyclical annealing: ramp up, then down, then up again
        cycle_length = total_epochs // 3
        cycle_position = epoch % cycle_length
        return min(1.0, 2 * cycle_position / cycle_length)
    
    elif method == 'sigmoid':
        # Sigmoid annealing for smooth transition
        return 1.0 / (1.0 + np.exp(-(epoch - total_epochs * 0.3) * 0.1))
    
    else:
        return 1.0

def temperature_annealing_schedule(epoch, initial_temp=2.0, final_temp=0.1):
    """
    Temperature annealing for discrete outputs.
    
    Args:
        epoch: Current epoch
        initial_temp: Starting temperature
        final_temp: Final temperature
        
    Returns:
        temperature: Current temperature
    """
    # Linear decrease from initial to final temperature
    return max(final_temp, initial_temp - (initial_temp - final_temp) * epoch / 100)
