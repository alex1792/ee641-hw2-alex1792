"""
Dataset loader for drum pattern generation task.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class DrumPatternDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize drum pattern dataset.
        
        Args:
            data_dir: Path to drum dataset directory
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load patterns from patterns.npz (faster loading)
        npz_path = os.path.join(data_dir, 'patterns.npz')
        data = np.load(npz_path)
        
        # Extract patterns and styles from NPZ
        # Keys: train_patterns, val_patterns, train_styles, val_styles
        self.patterns = data[f'{split}_patterns']
        self.styles = data[f'{split}_styles']
        
        # Load metadata from JSON (for instrument names, etc.)
        json_path = os.path.join(data_dir, 'patterns.json')
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        self.metadata = {
            'instruments': metadata['instruments'],
            'styles': metadata['styles'],
            'timesteps': metadata['timesteps']
        }
        self.instrument_names = metadata['instruments']
        self.style_names = metadata['styles']
    
        
        # Expected structure:
        # patterns: [N, 16, 9] binary arrays
        # self.patterns = data[split]['pattern']
        # styles: [N] style labels (0-4)
        # self.styles = data[split]['style_idx']
        # metadata: dict with instrument names, style names
        # self.metadata = {'instruments': data['instruments'], 'styles': data['styles']}
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        """
        Return a drum pattern sample.
        
        Returns:
            pattern: Binary tensor of shape [16, 9]
            style: Integer style label (0-4)
            density: Float indicating pattern density (for analysis)
        """
        pattern = self.patterns[idx]
        style = self.styles[idx]
        
        # Convert to tensor
        pattern_tensor = torch.from_numpy(pattern).float()
        
        # Compute density metric (fraction of active hits)
        density = pattern.sum() / (16 * 9)
        
        return pattern_tensor, style, density
    
    def pattern_to_pianoroll(self, pattern):
        """
        Convert pattern to visual piano roll representation.
        
        Args:
            pattern: Binary array [16, 9] or tensor
            
        Returns:
            pianoroll: Visual representation for plotting
        """
        if torch.is_tensor(pattern):
            pattern = pattern.cpu().numpy()
        
        # Create visual representation with instrument labels
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each active hit
        for t in range(16):
            for i in range(9):
                if pattern[t, i] > 0.5:
                    rect = patches.Rectangle((t, i), 1, 1, 
                                            linewidth=1, 
                                            edgecolor='black',
                                            facecolor='blue')
                    ax.add_patch(rect)
        
        # Add grid
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(17))
        ax.set_yticks(range(10))
        ax.set_yticklabels([''] + self.instrument_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Instrument')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        return fig