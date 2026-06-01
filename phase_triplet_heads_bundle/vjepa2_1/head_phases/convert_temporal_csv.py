"""
Convert temporal CSV format (path start end label) to simple format (path label)
"""

import pandas as pd

def convert_temporal_to_simple(input_csv, output_csv):
    """
    Convert from: path start_time end_time label
    To: path label
    
    Handles paths with embedded newlines (cleaning them)
    """
    print(f"Reading {input_csv}...")
    
    # Read file manually to handle newlines in paths
    samples = []
    with open(input_csv, 'r') as f:
        for line in f:
            # Remove newlines within the path and clean whitespace
            line = line.replace('\n', '').strip()
            parts = line.split()
            
            if len(parts) >= 4:
                # Last element is label, second-to-last and third-to-last are times
                # Everything before that is the path (which may contain spaces)
                label = parts[-1]
                end_time = parts[-2]
                start_time = parts[-3]
                video_path = ' '.join(parts[:-3])  # Rejoin path parts
                
                # Clean any remaining newlines/returns in path
                video_path = video_path.replace('\r', '').replace('\n', '')
                
                samples.append({
                    'video_path': video_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': label
                })
    
    df = pd.DataFrame(samples)
    
    # Explicitly convert labels to int64
    df['label'] = df['label'].astype('int64')
    
    print(f"Found {len(df)} samples")
    print(f"Unique videos: {df['video_path'].nunique()}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    print(f"Label dtype: {df['label'].dtype}")
    
    # Write simple 2-column format with explicit integer formatting
    with open(output_csv, 'w') as f:
        for _, row in df.iterrows():
            # Ensure label is written as pure integer without decimal point
            label_int = int(row['label'])
            f.write(f"{row['video_path']} {label_int}\n")
    
    print(f"\nCreated {output_csv}")
    
    # Show sample
    print("\nSample output (first 3 lines):")
    with open(output_csv, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.strip()}")
            else:
                break

if __name__ == "__main__":
    # Convert training data
    convert_temporal_to_simple(
        input_csv='app/csv_vjepa/surgical_phase_train.csv',
        output_csv='app/csv_vjepa/surg_phase_train.csv'
    )
    
    print("\n" + "="*60 + "\n")
    
    # Convert test data
    convert_temporal_to_simple(
        input_csv='app/csv_vjepa/surgical_phase_test.csv',
        output_csv='app/csv_vjepa/surg_phase_test.csv'
    )
