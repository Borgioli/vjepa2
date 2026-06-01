# Single-Head Multi-Task Classifier for V-JEPA 2

This implementation adds support for **single-head multi-task learning** to the triplet_recog_frozen evaluation framework.

## Architecture Overview

### Standard Multi-Head (Original)
```
Encoder → [Query1, Query2, Query3] → [Linear1, Linear2, Linear3] → [Task1, Task2, Task3]
```
- Each task has its own query token
- Each task has its own linear classifier
- Tasks learn separate representations

### Single-Head Multi-Task (New)
```
Encoder → [Single Query] → [Single Linear Layer] → [Task1 | Task2 | Task3]
```
- **One** query token for all tasks
- **One** linear layer that outputs concatenated predictions
- Tasks share the same representation
- Predictions are split by task based on num_classes_per_task

## Key Benefits

1. **Shared Representation**: Forces the model to learn a unified representation useful for all tasks
2. **Fewer Parameters**: Only 1 query token and 1 linear layer instead of N
3. **Better Regularization**: Shared features can improve generalization
4. **Simpler Architecture**: Easier to understand and debug

## Files Modified

### `/evals/triplet_recog_frozen/models.py`
- **Added**: `SingleHeadMultiTaskClassifier` class
  - Single `AttentivePooler` with `num_queries=1`
  - Single linear layer outputting `sum(num_classes_per_task)` logits
  - Splits output logits by task for loss computation

### `/evals/triplet_recog_frozen/eval.py`
- **Added**: Multi-task support in data configuration
  - New parameter: `num_classes_per_task` (list of integers)
  - New parameter: `use_single_head` (boolean)
- **Modified**: `run_one_epoch()` function
  - Handles multi-task labels from dataset
  - Computes separate loss per task
  - Tracks accuracy per task
  - Aggregates metrics across tasks

### `/evals/triplet_recog_frozen/example_config_multitask.yaml`
- **Created**: Example configuration showing how to use the new architecture

## Usage

### Configuration

```yaml
experiment:
  classifier:
    num_probe_blocks: 4
    num_heads: 16
    use_single_head: True  # Enable single-head architecture
  
  data:
    # Multi-task: specify classes per task
    num_classes_per_task: [7, 3]  # 7 phases, 3 actions
    
    # OR single-task (backward compatible):
    # num_classes: 7
```

### Dataset Requirements

For multi-task mode, your dataset must return multiple labels:

```python
class MultiTaskDataset:
    def __getitem__(self, idx):
        video = self.load_video(idx)
        phase_label = self.phase_labels[idx]
        action_label = self.action_labels[idx]
        
        # Return as list of labels
        return video, [phase_label, action_label]
```

### Running Training

```bash
# Use your existing training script with the new config
python app/main_distributed.py \
    --config configs/heads/phase_multitask.yaml
```

## Implementation Details

### Forward Pass

1. **Encoder**: Produces token representations `[B, T, D]`
2. **Pooler**: Single query attends to all tokens → `[B, 1, D]`
3. **Classifier**: Single linear layer → `[B, sum(num_classes_per_task)]`
4. **Split**: Logits split by task:
   - Task 0: logits[:, 0:7]
   - Task 1: logits[:, 7:10]

### Loss Computation

```python
# For each task, compute cross-entropy loss
losses = []
for task_idx in range(num_tasks):
    task_logits = output[f"task_{task_idx}"]
    task_label = labels_list[task_idx]
    task_loss = criterion(task_logits, task_label)
    losses.append(task_loss)

# Total loss is sum across tasks
total_loss = sum(losses)
```

### Accuracy Tracking

- Overall accuracy: Average across all tasks
- Per-task accuracy: Tracked separately and logged
- Example log output:
  ```
  [   10] 85.2% [82.1% 79.3%] loss: 0.421 [mem: 4.2e+03] | Tasks: T0=83.5% T1=81.0%
  ```

## Backward Compatibility

The implementation is **fully backward compatible**:

- Set `use_single_head: False` or omit it → Uses standard `AttentiveClassifier`
- Use `num_classes` instead of `num_classes_per_task` → Single-task mode
- Existing configs work without modification

## Comparison with Multi-Head Approach

| Feature | Multi-Head | Single-Head |
|---------|-----------|-------------|
| Query tokens | N (one per task) | 1 (shared) |
| Linear layers | N (one per task) | 1 (shared) |
| Representation | Task-specific | Shared |
| Parameters | More | Fewer |
| Use case | Tasks are very different | Tasks are related |

## Future Extensions

Possible enhancements:

1. **Task weighting**: Weight losses differently per task
2. **Auxiliary losses**: Add auxiliary prediction tasks
3. **Task-specific heads**: Shared pooler + separate linear layers
4. **Dynamic task selection**: Enable/disable tasks at runtime

## Example Workflows

### Surgical Phase + Action Recognition

```yaml
num_classes_per_task: [7, 3]  # 7 phases, 3 actions
```

### Surgical Phase + Tool Detection

```yaml
num_classes_per_task: [7, 5]  # 7 phases, 5 tools
```

### Triple Task

```yaml
num_classes_per_task: [7, 3, 5]  # phases, actions, tools
```

## Troubleshooting

### Issue: "IndexError: list index out of range"
- **Cause**: Dataset not returning correct number of labels
- **Fix**: Ensure dataset `__getitem__` returns list with length = `len(num_classes_per_task)`

### Issue: High loss on one task
- **Cause**: Class imbalance or task difficulty mismatch
- **Fix**: Consider task-specific loss weighting or separate learning rates

### Issue: Lower accuracy than multi-head
- **Cause**: Tasks may be too different to benefit from shared representation
- **Fix**: Use multi-head approach or add task-specific layers

## Citation

If you use this implementation, please cite the V-JEPA 2 paper:

```bibtex
@article{vjepa2,
  title={V-JEPA 2: ...},
  author={...},
  journal={...},
  year={2024}
}
```
