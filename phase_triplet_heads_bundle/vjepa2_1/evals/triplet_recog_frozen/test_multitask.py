#!/usr/bin/env python3
"""
Test script to verify the SingleHeadMultiTaskClassifier implementation.
This demonstrates the forward pass and output structure.
"""

import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.triplet_recog_frozen.models import SingleHeadMultiTaskClassifier


def test_single_head_multitask():
    """Test the single-head multi-task classifier."""
    
    print("="*80)
    print("Testing SingleHeadMultiTaskClassifier")
    print("="*80)
    
    # Configuration
    batch_size = 4
    num_tokens = 196  # e.g., 14x14 spatial tokens
    embed_dim = 1024
    num_heads = 16
    depth = 4
    
    # Multi-task: Tools (6) + Verbs (5) + Targets (10)
    num_classes_per_task = [6, 5, 10]
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of tokens: {num_tokens}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of probe blocks: {depth}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Tasks: {len(num_classes_per_task)}")
    print(f"  Classes per task: {num_classes_per_task}")
    print(f"  Total classes: {sum(num_classes_per_task)}")
    
    # Initialize classifier
    print(f"\nInitializing SingleHeadMultiTaskClassifier...")
    classifier = SingleHeadMultiTaskClassifier(
        num_classes_per_task=num_classes_per_task,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        use_activation_checkpointing=False,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create dummy input
    print(f"\nCreating dummy input: [{batch_size}, {num_tokens}, {embed_dim}]")
    x = torch.randn(batch_size, num_tokens, embed_dim)
    
    # Forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = classifier(x)
    
    # Check output structure
    print(f"\nOutput structure:")
    print(f"  Type: {type(output)}")
    print(f"  Keys: {list(output.keys())}")
    
    for task_idx, num_classes in enumerate(num_classes_per_task):
        task_key = f"task_{task_idx}"
        task_output = output[task_key]
        print(f"\n  {task_key}:")
        print(f"    Shape: {task_output.shape}")
        print(f"    Expected: [{batch_size}, {num_classes}]")
        print(f"    Match: {task_output.shape == torch.Size([batch_size, num_classes])}")
    
    # Verify shapes
    print(f"\n{'='*80}")
    print("Verification:")
    print(f"{'='*80}")
    
    all_correct = True
    for task_idx, num_classes in enumerate(num_classes_per_task):
        task_key = f"task_{task_idx}"
        expected_shape = torch.Size([batch_size, num_classes])
        actual_shape = output[task_key].shape
        is_correct = actual_shape == expected_shape
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} Task {task_idx}: {actual_shape} == {expected_shape}")
        all_correct = all_correct and is_correct
    
    if all_correct:
        print(f"\n{'='*80}")
        print("✓ All tests passed!")
        print(f"{'='*80}")
        return 0
    else:
        print(f"\n{'='*80}")
        print("✗ Some tests failed!")
        print(f"{'='*80}")
        return 1


def test_loss_computation():
    """Test loss computation with multi-task outputs."""
    
    print("\n" + "="*80)
    print("Testing Multi-Task Loss Computation")
    print("="*80)
    
    batch_size = 4
    num_classes_per_task = [6, 5, 10]
    
    # Create dummy predictions
    predictions = {
        "task_0": torch.randn(batch_size, num_classes_per_task[0]),
        "task_1": torch.randn(batch_size, num_classes_per_task[1]),
        "task_2": torch.randn(batch_size, num_classes_per_task[2]),
    }
    
    # Create dummy labels
    labels = [
        torch.randint(0, num_classes_per_task[0], (batch_size,)),
        torch.randint(0, num_classes_per_task[1], (batch_size,)),
        torch.randint(0, num_classes_per_task[2], (batch_size,)),
    ]
    
    print(f"\nPredictions:")
    for key, pred in predictions.items():
        print(f"  {key}: {pred.shape}")
    
    print(f"\nLabels:")
    for i, label in enumerate(labels):
        print(f"  task_{i}: {label.shape}")
    
    # Compute loss per task
    criterion = torch.nn.CrossEntropyLoss()
    task_losses = []
    
    print(f"\nComputing losses:")
    for task_idx in range(len(num_classes_per_task)):
        task_key = f"task_{task_idx}"
        task_loss = criterion(predictions[task_key], labels[task_idx])
        task_losses.append(task_loss)
        print(f"  {task_key} loss: {task_loss.item():.4f}")
    
    # Total loss
    total_loss = sum(task_losses)
    print(f"\nTotal loss (sum): {total_loss.item():.4f}")
    
    print(f"\n{'='*80}")
    print("✓ Loss computation test passed!")
    print(f"{'='*80}")
    
    return 0


def compare_architectures():
    """Compare single-head vs multi-head architectures."""
    
    print("\n" + "="*80)
    print("Architecture Comparison")
    print("="*80)
    
    embed_dim = 1024
    num_heads = 16
    depth = 4
    num_classes_per_task = [6, 5, 10]
    
    # Single-head
    single_head = SingleHeadMultiTaskClassifier(
        num_classes_per_task=num_classes_per_task,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
    )
    single_head_params = sum(p.numel() for p in single_head.parameters())
    
    print(f"\nSingle-Head Multi-Task:")
    print(f"  Pooler queries: 1")
    print(f"  Linear layers: 1")
    print(f"  Output size: {sum(num_classes_per_task)}")
    print(f"  Total parameters: {single_head_params:,}")
    
    # Simulate multi-head (separate classifiers)
    from src.models.attentive_pooler import AttentiveClassifier
    
    multi_head_params = 0
    for num_classes in num_classes_per_task:
        classifier = AttentiveClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            num_classes=num_classes,
        )
        multi_head_params += sum(p.numel() for p in classifier.parameters())
    
    print(f"\nMulti-Head (Separate Classifiers):")
    print(f"  Pooler queries: {len(num_classes_per_task)} (1 per task)")
    print(f"  Linear layers: {len(num_classes_per_task)} (1 per task)")
    print(f"  Total parameters: {multi_head_params:,}")
    
    print(f"\nParameter Reduction:")
    reduction = (1 - single_head_params / multi_head_params) * 100
    print(f"  Single-head uses {reduction:.1f}% fewer parameters")
    
    print(f"\n{'='*80}")
    
    return 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Single-Head Multi-Task Classifier Test Suite")
    print("="*80)
    
    try:
        # Run tests
        result1 = test_single_head_multitask()
        result2 = test_loss_computation()
        result3 = compare_architectures()
        
        # Summary
        print("\n" + "="*80)
        print("Test Suite Summary")
        print("="*80)
        
        if result1 == 0 and result2 == 0 and result3 == 0:
            print("✓ All tests passed successfully!")
            sys.exit(0)
        else:
            print("✗ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
