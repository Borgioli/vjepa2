"""
Convert surgical phase recognition data to VJEPA2 format.
Handles frame-level annotations and creates video clips.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import cv2
from tqdm import tqdm

# Configuration from your training script
BASE_DIR = Path("/path/to/surg_vid/probe_training/cholec_phases")
VIDEO_DIR = BASE_DIR / "videos"
ANNOTATION_DIR = BASE_DIR / "annotations/csv_output"
OUTPUT_DIR = Path("/path/to/vjepa2/app/csv_vjepa")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PHASE_MAPPING = {
    '1-exposure of the working area': 0,
    '2-retraction of the gallbladder neck': 1,
    '3-opening the anterior peritoneal layer of the triangle of calot': 2,
    '4-opening the posterior peritoneal layer of the triangle of calot': 3,
    '5-isolation of the cystic duct': 4,
    '6-isolation of the cystic artery': 5,
    '7-clipping of the cystic duct': 6,
    '8-clipping of the cystic artery': 7,
    '9-division of the cystic duct': 8,
    '10-dissection of the gallbladder from the liver': 9,
    '11-specimen retrival': 10,
}

def normalize_phase_name(phase_str):
    """Normalize phase name - from your code"""
    if not phase_str or pd.isna(phase_str):
        return None
    
    phase = str(phase_str).lower().strip()
    phase = phase.replace('retrieval', 'retrival')
    
    if '(' in phase:
        phase = phase.split('(')[0].strip()
    
    parts = phase.split('-')
    if len(parts) >= 2:
        phase_num = parts[0].strip()
        rest = '-'.join(parts[1:]).strip()
        
        if 'exposure' in rest or 'working area' in rest:
            return '1-exposure of the working area'
        elif 'retraction' in rest:
            return '2-retraction of the gallbladder neck'
        elif 'anterior' in rest and 'peritoneal' in rest:
            return '3-opening the anterior peritoneal layer of the triangle of calot'
        elif 'posterior' in rest and 'peritoneal' in rest:
            return '4-opening the posterior peritoneal layer of the triangle of calot'
        elif 'isolation' in rest and 'duct' in rest:
            return '5-isolation of the cystic duct'
        elif 'isolation' in rest and 'artery' in rest:
            return '6-isolation of the cystic artery'
        elif 'clipping' in rest and 'duct' in rest:
            return '7-clipping of the cystic duct'
        elif 'clipping' in rest and 'artery' in rest:
            return '8-clipping of the cystic artery'
        elif 'division' in rest and ('duct' in rest or 'artery' in rest):
            return '9-division of the cystic duct'
        elif 'dissection' in rest and 'gallbladder' in rest:
            return '10-dissection of the gallbladder from the liver'
        elif 'specimen' in rest or 'retrival' in rest or 'retrieval' in rest:
            return '11-specimen retrival'
    
    return None

def parse_time_to_seconds(time_str):
    """Convert time string to seconds - from your code"""
    if pd.isna(time_str) or time_str == '':
        return None
    
    time_str = str(time_str).strip()
    parts = time_str.split(':')
    
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        elif len(parts) == 4:
            hours, minutes, seconds, frames = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except:
        pass
    
    return None

def parse_time_range(time_range_str):
    """Parse time range string - from your code"""
    if pd.isna(time_range_str) or time_range_str == '':
        return None, None
    
    time_range_str = str(time_range_str).strip()
    
    if '-' in time_range_str:
        parts = time_range_str.split('-')
        start_time = parse_time_to_seconds(parts[0])
        end_time = parse_time_to_seconds(parts[1]) if len(parts) > 1 else None
        return start_time, end_time
    else:
        time = parse_time_to_seconds(time_range_str)
        return time, time

def load_annotations(annotation_path):
    """Load and parse annotation CSV - from your code"""
    df = pd.read_csv(annotation_path)
    
    annotations = []
    for _, row in df.iterrows():
        time_frame = row['TIME FRAME']
        phase = row['N.STEP']
        
        if pd.isna(phase) or phase == '' or 'ICG' in str(phase):
            continue
        
        normalized_phase = normalize_phase_name(phase)
        if normalized_phase is None or normalized_phase not in PHASE_MAPPING:
            continue
        
        start_time, end_time = parse_time_range(time_frame)
        
        if start_time is not None:
            annotations.append({
                'start_time': start_time,
                'end_time': end_time if end_time else start_time,
                'phase': normalized_phase,
                'phase_id': PHASE_MAPPING[normalized_phase]
            })
    
    return annotations

def get_video_duration(video_path):
    """Get video duration and FPS"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration, fps

def get_dominant_phase(annotations, start_time, end_time):
    """Get dominant phase in a time window"""
    phase_durations = {}
    
    for ann in annotations:
        overlap_start = max(start_time, ann['start_time'])
        overlap_end = min(end_time, ann['end_time'])
        
        if overlap_start < overlap_end:
            duration = overlap_end - overlap_start
            phase_id = ann['phase_id']
            phase_durations[phase_id] = phase_durations.get(phase_id, 0) + duration
    
    if not phase_durations:
        return -1
    
    return max(phase_durations.items(), key=lambda x: x[1])[0]

def create_vjepa_dataset(video_ids, clip_duration=5.0, overlap=2.5):
    """
    Create VJEPA2-compatible dataset with video clips.
    
    Args:
        video_ids: List of video IDs to process
        clip_duration: Duration of each clip in seconds (default 5s for ~60 frames at 30fps)
        overlap: Overlap between consecutive clips in seconds
    """
    samples = []
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        video_path = VIDEO_DIR / f"video_{video_id}.mp4"
        annotation_path = ANNOTATION_DIR / f"annotation_{video_id}.csv"
        
        if not video_path.exists() or not annotation_path.exists():
            print(f"Skipping video {video_id}: missing files")
            continue
        
        # Load annotations
        annotations = load_annotations(annotation_path)
        if not annotations:
            print(f"Skipping video {video_id}: no valid annotations")
            continue
        
        # Get video info
        duration, fps = get_video_duration(video_path)
        if duration <= 0:
            print(f"Skipping video {video_id}: invalid duration")
            continue
        
        # Create clips with sliding window
        step = clip_duration - overlap
        current_time = 0
        
        while current_time + clip_duration <= duration:
            end_time = current_time + clip_duration
            
            # Get dominant phase for this clip
            phase_id = get_dominant_phase(annotations, current_time, end_time)
            
            if phase_id >= 0:  # Valid phase
                samples.append({
                    'video_path': str(video_path),
                    'start_time': current_time,
                    'end_time': end_time,
                    'phase_id': phase_id,
                    'video_id': int(video_id)  # Convert to native Python int
                })
            
            current_time += step
    
    return samples

def main():
    print("Creating VJEPA2-compatible surgical phase recognition dataset")
    
    # Train/test split (from your script)
    all_video_ids = list(range(1, 47))
    np.random.seed(42)
    NUM_TRAIN_VIDEOS = 35
    train_ids = sorted(np.random.choice(all_video_ids, NUM_TRAIN_VIDEOS, replace=False))
    test_ids = sorted([vid for vid in all_video_ids if vid not in train_ids])
    
    # Convert numpy types to native Python types
    train_ids = [int(x) for x in train_ids]
    test_ids = [int(x) for x in test_ids]
    
    print(f"\nTrain videos ({len(train_ids)}): {train_ids}")
    print(f"Test videos ({len(test_ids)}): {test_ids}")
    
    # Create train dataset
    print("\nCreating training dataset...")
    train_samples = create_vjepa_dataset(train_ids, clip_duration=5.0, overlap=2.5)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_samples = create_vjepa_dataset(test_ids, clip_duration=5.0, overlap=2.5)
    
    print(f"\nTotal train samples: {len(train_samples)}")
    print(f"Total test samples: {len(test_samples)}")
    
    # Save in VJEPA2 format (space-separated: video_path label)
    train_csv = OUTPUT_DIR / "surgical_phase_train.csv"
    test_csv = OUTPUT_DIR / "surgical_phase_test.csv"
    
    with open(train_csv, 'w') as f:
        for sample in train_samples:
            # Format: video_path start_time end_time label
            # Ensure phase_id is written as integer (int64/long)
            phase_id = int(sample['phase_id'])
            f.write(f"{sample['video_path']} {sample['start_time']:.2f} {sample['end_time']:.2f} {phase_id}\n")
    
    with open(test_csv, 'w') as f:
        for sample in test_samples:
            # Ensure phase_id is written as integer (int64/long)
            phase_id = int(sample['phase_id'])
            f.write(f"{sample['video_path']} {sample['start_time']:.2f} {sample['end_time']:.2f} {phase_id}\n")
    
    # Save metadata
    metadata = {
        'num_classes': len(PHASE_MAPPING),
        'phase_mapping': PHASE_MAPPING,
        'train_videos': train_ids,
        'test_videos': test_ids,
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'clip_duration': 5.0,
        'overlap': 2.5
    }
    
    with open(OUTPUT_DIR / "surgical_phase_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created train CSV: {train_csv}")
    print(f"✓ Created test CSV: {test_csv}")
    print(f"✓ Created metadata: {OUTPUT_DIR / 'surgical_phase_metadata.json'}")
    
    # Print class distribution
    train_phases = [s['phase_id'] for s in train_samples]
    test_phases = [s['phase_id'] for s in test_samples]
    
    print("\nClass distribution (train):")
    for phase_id in range(len(PHASE_MAPPING)):
        count = train_phases.count(phase_id)
        print(f"  Phase {phase_id}: {count} samples")
    
    print("\nClass distribution (test):")
    for phase_id in range(len(PHASE_MAPPING)):
        count = test_phases.count(phase_id)
        print(f"  Phase {phase_id}: {count} samples")

if __name__ == '__main__':
    main()