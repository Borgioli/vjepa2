"""
Custom dataset for surgical phase recognition with temporal clips.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SurgicalPhaseDataset(Dataset):
    """
    Dataset for surgical phase recognition.
    Reads video clips based on start/end times.
    """
    
    def __init__(self, csv_path, frames_per_clip=64, transform=None):
        self.samples = []
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        
        # Parse CSV: video_path start_time end_time label
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    video_path, start_time, end_time, label = parts
                    self.samples.append({
                        'video_path': video_path,
                        'start_time': float(start_time),
                        'end_time': float(end_time),
                        'label': int(label)
                    })
        
        print(f"Loaded {len(self.samples)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video clip
        frames = self._load_video_clip(
            sample['video_path'],
            sample['start_time'],
            sample['end_time']
        )
        
        # Apply transform
        if self.transform:
            frames = self.transform(frames)
        
        label = sample['label']
        
        return frames, label
    
    def _load_video_clip(self, video_path, start_time, end_time):
        """Load frames from video between start_time and end_time"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            # Return dummy frames if video fails
            return torch.zeros(3, self.frames_per_clip, 224, 224)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        frames = []
        for i in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Sample frames to get exactly frames_per_clip
        if len(frames) == 0:
            return torch.zeros(3, self.frames_per_clip, 224, 224)
        
        # Uniform sampling
        indices = np.linspace(0, len(frames)-1, self.frames_per_clip, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        
        # Stack and convert to tensor [T, H, W, C]
        video_tensor = np.stack(sampled_frames, axis=0)
        
        return torch.from_numpy(video_tensor)