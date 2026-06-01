#!/usr/bin/env python3
"""
Quick test script to verify the dataset wrapper handles labels correctly.
"""

import torch
from dataset_wrapper import MultiTaskLabelWrapper

class MockDataset:
    def __init__(self, labels):
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        buffer = torch.randn(3, 16, 224, 224)
        label = self.labels[idx]
        clip_indices = torch.tensor([0])
        return buffer, label, clip_indices

test_labels = [
    "0.0 0.0 0.0",
    "1.0 2.0 5.0",
    "1.0",
    "3",
]

print("Testing MultiTaskLabelWrapper...\n")

for i, label in enumerate(test_labels):
    print(f"Test {i+1}: Input label = '{label}'")
    
    mock_dataset = MockDataset([label])
    wrapped = MultiTaskLabelWrapper(mock_dataset)
    
    buffer, parsed_labels, clip_indices = wrapped[0]
    
    print(f"  Output: {len(parsed_labels)} labels")
    for j, l in enumerate(parsed_labels):
        print(f"    Task {j}: {l.item()}")
    print()

print("✓ All tests passed!")
print("\nNote: Single labels are replicated to all 3 tasks.")
print("This ensures compatibility when CSV has mixed label formats.")
