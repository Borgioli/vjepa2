#!/usr/bin/env python3
"""
Standalone test for missing label handling in MultiTaskLabelWrapper.
Run directly without pytest: python test_missing_labels.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class MockTensor:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"MockTensor({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, MockTensor):
            return self.value == other.value
        return False
    
    def clone(self):
        return MockTensor(self.value)

class MockTorch:
    @staticmethod
    def tensor(value):
        return MockTensor(value)
    
    @staticmethod
    def is_tensor(obj):
        return isinstance(obj, MockTensor)

sys.modules['torch'] = MockTorch()
sys.modules['torch.utils'] = type(sys)('torch.utils')
sys.modules['torch.utils.data'] = type(sys)('torch.utils.data')

class Dataset:
    pass

sys.modules['torch.utils.data'].Dataset = Dataset

from evals.triplet_recog_frozen.dataset_wrapper import MultiTaskLabelWrapper

def test_missing_labels():
    """Test that missing/empty labels return empty list"""
    
    class MockDataset:
        def __len__(self):
            return 3
        
        def __getitem__(self, index):
            if index == 0:
                return "video_data", "1.0 2.0 3.0", "clip_indices"
            elif index == 1:
                return "video_data", "", "clip_indices"
            elif index == 2:
                return "video_data", "   ", "clip_indices"
    
    wrapper = MultiTaskLabelWrapper(MockDataset())
    
    print("Testing MultiTaskLabelWrapper with missing labels...\n")
    
    print("Test 1: Valid labels '1.0 2.0 3.0'")
    buffer, labels, clip_indices = wrapper[0]
    print(f"  Result: {len(labels)} labels")
    assert len(labels) == 3, f"Expected 3 labels, got {len(labels)}"
    assert labels[0].value == 1, f"Expected first label to be 1, got {labels[0].value}"
    assert labels[1].value == 2, f"Expected second label to be 2, got {labels[1].value}"
    assert labels[2].value == 3, f"Expected third label to be 3, got {labels[2].value}"
    print("  ✓ PASSED\n")
    
    print("Test 2: Empty string ''")
    buffer, labels, clip_indices = wrapper[1]
    print(f"  Result: {len(labels)} labels (empty list)")
    assert len(labels) == 0, f"Expected 0 labels (empty list), got {len(labels)}"
    print("  ✓ PASSED\n")
    
    print("Test 3: Whitespace only '   '")
    buffer, labels, clip_indices = wrapper[2]
    print(f"  Result: {len(labels)} labels (empty list)")
    assert len(labels) == 0, f"Expected 0 labels (empty list), got {len(labels)}"
    print("  ✓ PASSED\n")
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nBehavior summary:")
    print("- Valid labels (e.g., '1.0 2.0 3.0') → parsed into 3 tensors")
    print("- Empty string ('') → returns empty list []")
    print("- Whitespace only ('   ') → returns empty list []")
    print("\nThis means the dataloader will 'not load' missing labels,")
    print("and the training loop will skip loss/backprop for those samples.")

if __name__ == "__main__":
    try:
        test_missing_labels()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
