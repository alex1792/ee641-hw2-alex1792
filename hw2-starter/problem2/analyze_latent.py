"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
from pathlib import Path

def visualize_latent_hierarchy(model, data_loader, device='cuda', save_dir='results/latent_analysis'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    model.eval()
    
    # Collect all latent representations
    z_high_list = []
    z_low_list = []
    styles_list = []
    patterns_list = []
    
    with torch.no_grad():
        for patterns, styles, densities in data_loader:
            patterns = patterns.to(device)
            
            # Encode to get latent representations
            z_high, z_low, mu_high, logvar_high, mu_low, logvar_low = model.encode_hierarchy(patterns)
            
            z_high_list.append(z_high.cpu().numpy())
            z_low_list.append(z_low.cpu().numpy())
            styles_list.append(styles.numpy())
            patterns_list.append(patterns.cpu().numpy())
    
    # Concatenate all batches
    z_high_all = np.concatenate(z_high_list, axis=0)
    z_low_all = np.concatenate(z_low_list, axis=0)
    styles_all = np.concatenate(styles_list, axis=0)
    patterns_all = np.concatenate(patterns_list, axis=0)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. t-SNE visualization of z_high space
    print("Computing t-SNE for z_high space...")
    tsne_high = TSNE(n_components=2, random_state=42, perplexity=30)
    z_high_2d = tsne_high.fit_transform(z_high_all)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(z_high_2d[:, 0], z_high_2d[:, 1], c=styles_all, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of z_high (Style) Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Add style labels
    style_names = ['rock', 'jazz', 'hiphop', 'electronic', 'latin']
    for i, style in enumerate(style_names):
        mask = styles_all == i
        if np.any(mask):
            center = np.mean(z_high_2d[mask], axis=0)
            plt.annotate(style, center, fontsize=12, ha='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/z_high_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE visualization of z_low space
    print("Computing t-SNE for z_low space...")
    tsne_low = TSNE(n_components=2, random_state=42, perplexity=30)
    z_low_2d = tsne_low.fit_transform(z_low_all)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(z_low_2d[:, 0], z_low_2d[:, 1], c=styles_all, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of z_low (Variation) Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/z_low_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Hierarchical visualization: z_high clusters with z_low variations
    plt.figure(figsize=(16, 12))
    
    # Find cluster centers in z_high space
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(z_high_all)
    cluster_centers = kmeans.cluster_centers_
    
    for cluster_idx in range(5):
        plt.subplot(2, 3, cluster_idx + 1)
        
        # Find points in this cluster
        cluster_mask = cluster_labels == cluster_idx
        
        if np.any(cluster_mask):
            # Get corresponding z_low points
            cluster_z_low = z_low_all[cluster_mask]
            cluster_styles = styles_all[cluster_mask]
            
            # Plot z_low variations for this cluster
            scatter = plt.scatter(cluster_z_low[:, 0], cluster_z_low[:, 1], 
                                c=cluster_styles, cmap='tab10', alpha=0.7, s=20)
            
            plt.title(f'z_low variations in cluster {cluster_idx}')
            plt.xlabel('z_low[0]')
            plt.ylabel('z_low[1]')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hierarchical_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent hierarchy visualizations saved to {save_dir}/")
    
    return {
        'z_high_all': z_high_all,
        'z_low_all': z_low_all,
        'styles_all': styles_all,
        'patterns_all': patterns_all,
        'z_high_2d': z_high_2d,
        'z_low_2d': z_low_2d
    }

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda', save_dir='results/generated_patterns'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Ensure patterns are on the correct device and have batch dimension
        pattern1 = pattern1.unsqueeze(0).to(device) if pattern1.dim() == 2 else pattern1.to(device)
        pattern2 = pattern2.unsqueeze(0).to(device) if pattern2.dim() == 2 else pattern2.to(device)
        
        # Encode both patterns
        z_high1, z_low1, _, _, _, _ = model.encode_hierarchy(pattern1)
        z_high2, z_low2, _, _, _, _ = model.encode_hierarchy(pattern2)
        
        # Linear interpolation in latent space
        interpolations = []
        
        for i in range(n_steps + 1):
            alpha = i / n_steps
            
            # Interpolate both latent levels
            z_high_interp = (1 - alpha) * z_high1 + alpha * z_high2
            z_low_interp = (1 - alpha) * z_low1 + alpha * z_low2
            
            # Decode interpolated latents
            pattern_interp = model.decode_hierarchy(z_high_interp, z_low_interp)
            pattern_interp = torch.sigmoid(pattern_interp)
            
            interpolations.append(pattern_interp.cpu().squeeze(0))
    
    # Visualize interpolation sequence
    fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(20, 6))
    axes = axes.flatten()
    
    for i, pattern in enumerate(interpolations):
        if i < len(axes):
            # Create piano roll visualization
            pattern_np = pattern.numpy()
            
            # Plot drum pattern
            im = axes[i].imshow(pattern_np.T, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Step {i}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Instrument')
            
            # Set instrument labels
            instrument_names = ['kick', 'snare', 'hihat_closed', 'hihat_open',
                              'tom_low', 'tom_high', 'crash', 'ride', 'clap']
            axes[i].set_yticks(range(9))
            axes[i].set_yticklabels(instrument_names, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/style_interpolation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual patterns
    for i, pattern in enumerate(interpolations):
        np.save(f'{save_dir}/interpolation_step_{i:02d}.npy', pattern.numpy())
    
    print(f"Style interpolation saved to {save_dir}/")
    
    return interpolations

def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    model.eval()
    
    # Collect latent representations grouped by style
    style_latents = {i: {'z_high': [], 'z_low': []} for i in range(5)}
    
    with torch.no_grad():
        for patterns, styles, densities in data_loader:
            patterns = patterns.to(device)
            
            z_high, z_low, _, _, _, _ = model.encode_hierarchy(patterns)
            
            for i, style in enumerate(styles):
                style_latents[style.item()]['z_high'].append(z_high[i].cpu().numpy())
                style_latents[style.item()]['z_low'].append(z_low[i].cpu().numpy())
    
    # Convert to numpy arrays
    for style in style_latents:
        style_latents[style]['z_high'] = np.array(style_latents[style]['z_high'])
        style_latents[style]['z_low'] = np.array(style_latents[style]['z_low'])
    
    # Compute disentanglement metrics
    style_names = ['rock', 'jazz', 'hiphop', 'electronic', 'latin']
    
    # 1. z_high within-style vs between-style variance
    z_high_within_var = []
    z_high_between_centers = []
    
    for style in range(5):
        if len(style_latents[style]['z_high']) > 0:
            z_high_style = style_latents[style]['z_high']
            within_var = np.mean(np.var(z_high_style, axis=0))
            z_high_within_var.append(within_var)
            
            center = np.mean(z_high_style, axis=0)
            z_high_between_centers.append(center)
    
    z_high_between_var = np.var(z_high_between_centers, axis=0).mean()
    z_high_within_var_avg = np.mean(z_high_within_var)
    
    # 2. z_low within-style variance (should be high for good variation)
    z_low_within_var = []
    for style in range(5):
        if len(style_latents[style]['z_low']) > 0:
            z_low_style = style_latents[style]['z_low']
            within_var = np.mean(np.var(z_low_style, axis=0))
            z_low_within_var.append(within_var)
    
    z_low_within_var_avg = np.mean(z_low_within_var)
    
    # 3. Disentanglement score
    disentanglement_score = z_high_between_var / (z_high_between_var + z_high_within_var_avg)
    
    results = {
        'z_high_between_style_variance': z_high_between_var,
        'z_high_within_style_variance': z_high_within_var_avg,
        'z_low_within_style_variance': z_low_within_var_avg,
        'disentanglement_score': disentanglement_score,
        'style_latents': style_latents
    }
    
    # Print results
    print("Disentanglement Analysis:")
    print(f"z_high between-style variance: {z_high_between_var:.4f}")
    print(f"z_high within-style variance: {z_high_within_var_avg:.4f}")
    print(f"z_low within-style variance: {z_low_within_var_avg:.4f}")
    print(f"Disentanglement score: {disentanglement_score:.4f}")
    print("(Higher score indicates better style separation)")
    
    return results

def controllable_generation(model, genre_labels, device='cuda', save_dir='results/generated_patterns'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    style_names = ['rock', 'jazz', 'hiphop', 'electronic', 'latin']
    generated_patterns = {}
    
    with torch.no_grad():
        for style_idx, style_name in enumerate(style_names):
            generated_patterns[style_name] = []
            
            # Generate multiple samples for this style
            for sample_idx in range(10):
                # Sample z_high from prior (for style)
                z_high = torch.randn(1, model.z_high_dim).to(device)
                
                # Sample z_low from prior (for variation)
                z_low = torch.randn(1, model.z_low_dim).to(device)
                
                # Decode to pattern
                pattern = model.decode_hierarchy(z_high, z_low, temperature=0.8)
                pattern = torch.sigmoid(pattern)
                
                generated_patterns[style_name].append(pattern.cpu().squeeze(0))
                
                # Save individual pattern
                np.save(f'{save_dir}/{style_name}_sample_{sample_idx:02d}.npy', 
                       pattern.cpu().squeeze(0).numpy())
    
    # Create visualization of generated patterns
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    
    for style_idx, style_name in enumerate(style_names):
        for sample_idx in range(10):
            pattern = generated_patterns[style_name][sample_idx]
            pattern_np = pattern.numpy()
            
            # Plot pattern
            im = axes[style_idx, sample_idx].imshow(pattern_np.T, cmap='Blues', aspect='auto')
            axes[style_idx, sample_idx].set_title(f'{style_name} {sample_idx+1}', fontsize=8)
            axes[style_idx, sample_idx].set_xticks([])
            axes[style_idx, sample_idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/generated_patterns_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated patterns saved to {save_dir}/")
    
    return generated_patterns

def analyze_latent_dimensions(model, data_loader, device='cuda', save_dir='results/latent_analysis'):
    """
    Analyze what each latent dimension controls.
    """
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect latent representations
    z_high_list = []
    z_low_list = []
    densities_list = []
    
    with torch.no_grad():
        for patterns, styles, densities in data_loader:
            patterns = patterns.to(device)
            
            z_high, z_low, _, _, _, _ = model.encode_hierarchy(patterns)
            
            z_high_list.append(z_high.cpu().numpy())
            z_low_list.append(z_low.cpu().numpy())
            densities_list.append(densities.numpy())
    
    z_high_all = np.concatenate(z_high_list, axis=0)
    z_low_all = np.concatenate(z_low_list, axis=0)
    densities_all = np.concatenate(densities_list, axis=0)
    
    # Get actual dimensions
    actual_z_high_dim = z_high_all.shape[1]
    actual_z_low_dim = z_low_all.shape[1]
    
    # Analyze correlations between latent dimensions and pattern density
    print("Latent Dimension Analysis:")
    print("z_high dimensions vs pattern density:")
    for i in range(actual_z_high_dim):
        corr = np.corrcoef(z_high_all[:, i], densities_all)[0, 1]
        print(f"  z_high[{i}]: correlation = {corr:.4f}")
    
    print("z_low dimensions vs pattern density:")
    for i in range(actual_z_low_dim):
        corr = np.corrcoef(z_low_all[:, i], densities_all)[0, 1]
        print(f"  z_low[{i}]: correlation = {corr:.4f}")
    
    # Visualize latent dimension distributions
    max_dims = max(actual_z_high_dim, actual_z_low_dim)
    
    # Create figure with appropriate number of columns
    fig, axes = plt.subplots(2, max_dims, figsize=(4*max_dims, 8))
    
    # Handle case where we have only one column
    if max_dims == 1:
        axes = axes.reshape(2, 1)
    
    # z_high dimensions
    for i in range(actual_z_high_dim):
        axes[0, i].hist(z_high_all[:, i], bins=50, alpha=0.7)
        axes[0, i].set_title(f'z_high[{i}]')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
    
    # Hide unused z_high subplots
    for i in range(actual_z_high_dim, max_dims):
        axes[0, i].set_visible(False)
    
    # z_low dimensions
    for i in range(actual_z_low_dim):
        axes[1, i].hist(z_low_all[:, i], bins=50, alpha=0.7)
        axes[1, i].set_title(f'z_low[{i}]')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    # Hide unused z_low subplots
    for i in range(actual_z_low_dim, max_dims):
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent dimension analysis saved to {save_dir}/")