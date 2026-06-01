# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset wrapper to convert string labels to multi-task format.
This wraps the VideoDataset to parse triplet labels from strings like "0.0 0.0 0.0"
into separate tensors for each task.
"""

import csv

import torch
from torch.utils.data import Dataset


def parse_triplet_label(label):
    """
    Parse a triplet-style label into [tool, verb, target] tensors.

    Supports strings such as "0 1 2" and keeps the previous single-value fallback
    behavior for compatibility with older paths.
    """

    import math

    if isinstance(label, float) and math.isnan(label):
        raise ValueError("Label is NaN - this row should have been filtered out during dataset loading")

    if isinstance(label, (list, tuple)):
        parsed = [torch.tensor(l) if not torch.is_tensor(l) else l for l in label]
        if len(parsed) == 0:
            return []
        while len(parsed) < 3:
            parsed.append(parsed[0] if len(parsed) > 0 else torch.tensor(0))
        return parsed[:3]

    if torch.is_tensor(label):
        return [label.clone(), label.clone(), label.clone()]

    if isinstance(label, str):
        label = label.strip('"').strip("'").strip()
        if label == "":
            return []

        parts = label.split()

        if len(parts) == 3:
            try:
                tool = int(float(parts[0]))
                verb = int(float(parts[1]))
                target = int(float(parts[2]))
                return [torch.tensor(tool), torch.tensor(verb), torch.tensor(target)]
            except (ValueError, IndexError) as exc:
                if any("nan" in str(part).lower() for part in parts):
                    raise ValueError(
                        f"Label contains NaN value - this row should have been filtered: '{label}'"
                    ) from exc
                raise ValueError(f"Could not parse label '{label}' into three integers: {exc}") from exc
        if len(parts) == 1:
            try:
                single_label = int(float(parts[0]))
                return [torch.tensor(single_label), torch.tensor(single_label), torch.tensor(single_label)]
            except (ValueError, IndexError) as exc:
                if "nan" in str(parts[0]).lower():
                    raise ValueError(
                        f"Label contains NaN value - this row should have been filtered: '{label}'"
                    ) from exc
                raise ValueError(f"Could not parse single label '{label}': {exc}") from exc
        if len(parts) == 0:
            return []
        raise ValueError(f"Expected 1 or 3 space-separated values in label, got {len(parts)}: '{label}'")

    try:
        single_label = int(float(label))
        return [torch.tensor(single_label), torch.tensor(single_label), torch.tensor(single_label)]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported label type or format: {type(label)} - {label}") from exc


class MultiTaskLabelWrapper(Dataset):
    """
    Wraps a VideoDataset to convert string labels into multi-task format.
    
    Expected label format in CSV: "tool_id verb_id target_id"
    Example: "0.0 0.0 0.0" or "1 2 5"
    
    Returns labels as: [tool_tensor, verb_tensor, target_tensor]
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: The underlying VideoDataset instance
        """
        super().__init__()
        object.__setattr__(self, 'base_dataset', base_dataset)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        result = self.base_dataset[index]
        
        if len(result) == 3:
            buffer, label, clip_indices = result
        else:
            return result
        
        parsed_labels = self._parse_label(label)
        
        return buffer, parsed_labels, clip_indices
    
    def _parse_label(self, label):
        """
        Parse label string into list of tensors.
        
        Args:
            label: String like "0.0 0.0 0.0" or "1 2 5" or already a list/tensor
            
        Returns:
            List of three tensors: [tool_label, verb_label, target_label]
        """
        return parse_triplet_label(label)
    
    def __getattr__(self, name):
        try:
            base_dataset = object.__getattribute__(self, 'base_dataset')
            return getattr(base_dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ConditionedTaskFromTripletWrapper(Dataset):
    """
    Wrap a triplet-labeled dataset and expose one task label plus one or more
    conditioning labels.

    Example use cases:
    - input: clip + tool label, output: verb label
    - input: clip + tool + verb labels, output: target label
    """

    def __init__(self, base_dataset, conditioning_task_idx=0, target_task_idx=1):
        super().__init__()
        object.__setattr__(self, "base_dataset", base_dataset)
        object.__setattr__(self, "target_task_idx", int(target_task_idx))
        if isinstance(conditioning_task_idx, (list, tuple)):
            conditioning_task_indices = tuple(int(task_idx) for task_idx in conditioning_task_idx)
        else:
            conditioning_task_indices = (int(conditioning_task_idx),)
        if len(conditioning_task_indices) == 0:
            raise ValueError("conditioning_task_idx must contain at least one task index")
        object.__setattr__(self, "conditioning_task_indices", conditioning_task_indices)

        for attr_name in ("target_task_idx",):
            attr_value = object.__getattribute__(self, attr_name)
            if attr_value < 0 or attr_value > 2:
                raise ValueError(f"{attr_name} must be in [0, 2], got {attr_value}")
        for task_idx in conditioning_task_indices:
            if task_idx < 0 or task_idx > 2:
                raise ValueError(f"conditioning task indices must be in [0, 2], got {task_idx}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]
        if len(result) != 3:
            raise ValueError(
                "ConditionedTaskFromTripletWrapper expects the base dataset to return "
                "(buffer, label, clip_indices)."
            )

        buffer, label, clip_indices = result
        parsed_labels = parse_triplet_label(label)
        if len(parsed_labels) != 3:
            raise ValueError(f"Expected a triplet label with 3 values, got: {label}")

        target_label = parsed_labels[self.target_task_idx]
        conditioning_values = [parsed_labels[task_idx] for task_idx in self.conditioning_task_indices]
        if len(conditioning_values) == 1:
            conditioning_label = conditioning_values[0]
        else:
            conditioning_label = torch.stack([value.to(dtype=torch.long) for value in conditioning_values], dim=0)
        return buffer, [target_label], clip_indices, conditioning_label

    def __getattr__(self, name):
        try:
            base_dataset = object.__getattribute__(self, "base_dataset")
            return getattr(base_dataset, name)
        except AttributeError as exc:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from exc


class MultiLabelTaskLabelWrapper(Dataset):
    """
    Wraps a VideoDataset that returns sample paths and injects grouped multi-label targets.

    Expected lookup CSV format is the sidecar dataset produced by
    build_triplet_multilabel_dataset.py, with one row per clip_path and one or more
    space-delimited multi-hot columns such as tool_multihot, verb_multihot, etc.
    """

    def __init__(self, base_dataset, multilabel_csv_path, label_fields, count_source_fields=None):
        super().__init__()
        object.__setattr__(self, "base_dataset", base_dataset)
        object.__setattr__(self, "label_fields", list(label_fields))
        if count_source_fields is None:
            count_source_fields = []
        object.__setattr__(self, "count_source_fields", [str(field) for field in count_source_fields])
        object.__setattr__(self, "label_lookup", self._load_label_lookup(multilabel_csv_path))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]

        if len(result) != 4:
            raise ValueError(
                "MultiLabelTaskLabelWrapper expects the base dataset to return "
                "(buffer, label, clip_indices, sample_path)."
            )

        buffer, _, clip_indices, sample_path = result
        sample_path = str(sample_path)
        if sample_path not in self.label_lookup:
            raise KeyError(f"Could not find multilabel annotations for sample path: {sample_path}")

        labels = [label.clone() for label in self.label_lookup[sample_path]]
        return buffer, labels, clip_indices

    def _parse_multihot(self, value):
        value = str(value).strip()
        if value == "":
            raise ValueError("Encountered empty multi-hot label field in multilabel dataset")
        return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)

    def _load_label_lookup(self, multilabel_csv_path):
        label_lookup = {}
        source_field_to_idx = {field: idx for idx, field in enumerate(self.label_fields)}
        missing_count_sources = [
            field for field in self.count_source_fields if field not in source_field_to_idx
        ]
        if missing_count_sources:
            raise ValueError(
                "count_source_fields must refer to entries in label_fields. "
                f"Missing: {missing_count_sources}"
            )
        with open(multilabel_csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_fields = [field for field in ["clip_path", *self.label_fields] if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in multilabel CSV {multilabel_csv_path}: {missing_fields}"
                )

            for row in reader:
                clip_path = str(row["clip_path"]).strip()
                if not clip_path:
                    continue
                labels = [self._parse_multihot(row[field]) for field in self.label_fields]
                for count_source_field in self.count_source_fields:
                    source_label = labels[source_field_to_idx[count_source_field]]
                    labels.append(torch.tensor(int(source_label.sum().item()), dtype=torch.long))
                label_lookup[clip_path] = labels
        return label_lookup

    def __getattr__(self, name):
        try:
            base_dataset = object.__getattribute__(self, "base_dataset")
            return getattr(base_dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ConditionedMultiLabelTaskLabelWrapper(Dataset):
    """
    Wrap a VideoDataset and expose grouped multi-label targets plus a discrete
    conditioning label or labels from the runtime CSV.

    This is useful for datasets with repeated clip paths, such as one sample per
    (clip, tool) pair where the label is an action multi-hot for that tool, or
    one sample per (clip, tool, verb) row for target multi-hot labels.
    """

    def __init__(self, base_dataset, multilabel_csv_path, label_fields, condition_field):
        super().__init__()
        object.__setattr__(self, "base_dataset", base_dataset)
        object.__setattr__(self, "label_fields", list(label_fields))
        if isinstance(condition_field, (list, tuple)):
            condition_fields = tuple(str(field) for field in condition_field)
        else:
            condition_fields = (str(condition_field),)
        if len(condition_fields) == 0:
            raise ValueError("condition_field must contain at least one field")
        object.__setattr__(self, "condition_fields", condition_fields)
        object.__setattr__(self, "condition_field", condition_fields[0])
        object.__setattr__(self, "label_lookup", self._load_label_lookup(multilabel_csv_path))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]

        if len(result) != 4:
            raise ValueError(
                "ConditionedMultiLabelTaskLabelWrapper expects the base dataset to return "
                "(buffer, label, clip_indices, sample_path)."
            )

        buffer, condition_label, clip_indices, sample_path = result
        sample_path = str(sample_path)
        condition_label = self._parse_condition_label(condition_label)
        lookup_key = (sample_path, *self._condition_label_to_tuple(condition_label))
        if lookup_key not in self.label_lookup:
            raise KeyError(
                "Could not find conditioned multilabel annotations for "
                f"sample path/condition: {lookup_key}"
            )

        labels = [label.clone() for label in self.label_lookup[lookup_key]]
        return buffer, labels, clip_indices, condition_label

    def _parse_condition_label(self, value):
        num_condition_fields = len(self.condition_fields)
        if torch.is_tensor(value):
            condition_label = value.clone().to(dtype=torch.long)
            if num_condition_fields == 1:
                return condition_label.reshape(-1)[0]
            condition_label = condition_label.reshape(-1)
            if condition_label.numel() != num_condition_fields:
                raise ValueError(
                    f"Expected {num_condition_fields} conditioning labels, got tensor "
                    f"with shape {tuple(value.shape)}"
                )
            return condition_label
        value = str(value).strip().strip('"').strip("'")
        if value == "":
            raise ValueError("Encountered empty conditioning label in runtime dataset")
        tokens = value.split()
        if len(tokens) != num_condition_fields:
            raise ValueError(
                f"Expected {num_condition_fields} conditioning labels from fields "
                f"{self.condition_fields}, got {len(tokens)} token(s): {value!r}"
            )
        parsed = [int(float(token)) for token in tokens]
        if num_condition_fields == 1:
            return torch.tensor(parsed[0], dtype=torch.long)
        return torch.tensor(parsed, dtype=torch.long)

    def _condition_label_to_tuple(self, condition_label):
        condition_label = condition_label.reshape(-1).to(dtype=torch.long)
        if condition_label.numel() != len(self.condition_fields):
            raise ValueError(
                f"Expected {len(self.condition_fields)} conditioning labels, "
                f"got {condition_label.numel()}"
            )
        return tuple(int(value.item()) for value in condition_label)

    def _parse_multihot(self, value):
        value = str(value).strip()
        if value == "":
            raise ValueError("Encountered empty multi-hot label field in conditioned multilabel dataset")
        return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)

    def _load_label_lookup(self, multilabel_csv_path):
        label_lookup = {}
        required_fields = ["clip_path", *self.condition_fields, *self.label_fields]
        with open(multilabel_csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_fields = [field for field in required_fields if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in conditioned multilabel CSV {multilabel_csv_path}: {missing_fields}"
                )

            for row in reader:
                clip_path = str(row["clip_path"]).strip()
                if not clip_path:
                    continue
                condition_values = tuple(
                    int(float(str(row[field]).strip())) for field in self.condition_fields
                )
                lookup_key = (clip_path, *condition_values)
                if lookup_key in label_lookup:
                    raise ValueError(
                        f"Duplicate conditioned multilabel row for clip/condition: {lookup_key}"
                    )
                label_lookup[lookup_key] = [
                    self._parse_multihot(row[field]) for field in self.label_fields
                ]
        return label_lookup

    def __getattr__(self, name):
        try:
            base_dataset = object.__getattribute__(self, "base_dataset")
            return getattr(base_dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class MultiConditionMultiLabelTaskLabelWrapper(Dataset):
    """
    Wrap a VideoDataset and expose multiple conditioning rows per clip.

    The lookup CSV has one row per clip. Each condition field is a space-delimited
    list, and each label field is a semicolon-delimited list of multi-hot vectors
    aligned with those condition rows.
    """

    def __init__(self, base_dataset, multilabel_csv_path, label_fields, condition_fields):
        super().__init__()
        object.__setattr__(self, "base_dataset", base_dataset)
        object.__setattr__(self, "label_fields", list(label_fields))
        if isinstance(condition_fields, (list, tuple)):
            normalized_condition_fields = tuple(str(field) for field in condition_fields)
        else:
            normalized_condition_fields = (str(condition_fields),)
        if len(normalized_condition_fields) == 0:
            raise ValueError("condition_fields must contain at least one field")
        object.__setattr__(self, "condition_fields", normalized_condition_fields)
        object.__setattr__(self, "label_lookup", self._load_label_lookup(multilabel_csv_path))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        result = self.base_dataset[index]

        if len(result) != 4:
            raise ValueError(
                "MultiConditionMultiLabelTaskLabelWrapper expects the base dataset to return "
                "(buffer, label, clip_indices, sample_path)."
            )

        buffer, _, clip_indices, sample_path = result
        sample_path = str(sample_path)
        if sample_path not in self.label_lookup:
            raise KeyError(
                "Could not find multi-condition multilabel annotations for "
                f"sample path: {sample_path}"
            )

        condition_labels, labels = self.label_lookup[sample_path]
        condition_labels = condition_labels.clone()
        labels = [label.clone() for label in labels]
        condition_mask = torch.ones(condition_labels.shape[0], dtype=torch.bool)
        return buffer, labels, clip_indices, condition_labels, condition_mask

    def _parse_condition_values(self, value):
        value = str(value).strip()
        if value == "":
            raise ValueError("Encountered empty conditioning field in multi-condition dataset")
        return [int(float(token)) for token in value.split()]

    def _parse_multihot_sequence(self, value):
        value = str(value).strip()
        if value == "":
            raise ValueError("Encountered empty multi-hot label sequence in multi-condition dataset")
        vectors = []
        for vector in value.split(";"):
            vector = vector.strip()
            if not vector:
                continue
            vectors.append(torch.tensor([float(token) for token in vector.split()], dtype=torch.float32))
        if not vectors:
            raise ValueError("Encountered multi-hot label sequence with no usable vectors")
        first_shape = vectors[0].shape
        for vector in vectors:
            if vector.shape != first_shape:
                raise ValueError(
                    "All multi-hot vectors in a sequence must have the same shape, "
                    f"got {vector.shape} and {first_shape}"
                )
        return torch.stack(vectors, dim=0)

    def _load_label_lookup(self, multilabel_csv_path):
        label_lookup = {}
        required_fields = ["clip_path", *self.condition_fields, *self.label_fields]
        with open(multilabel_csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_fields = [field for field in required_fields if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in multi-condition multilabel CSV "
                    f"{multilabel_csv_path}: {missing_fields}"
                )

            for row in reader:
                clip_path = str(row["clip_path"]).strip()
                if not clip_path:
                    continue

                condition_columns = [
                    self._parse_condition_values(row[field]) for field in self.condition_fields
                ]
                num_condition_rows = len(condition_columns[0])
                if num_condition_rows == 0:
                    raise ValueError(f"No conditioning rows found for clip: {clip_path}")
                for condition_values in condition_columns:
                    if len(condition_values) != num_condition_rows:
                        raise ValueError(
                            "Condition fields must have aligned lengths for clip "
                            f"{clip_path}: {[len(values) for values in condition_columns]}"
                        )

                condition_labels = torch.tensor(
                    list(zip(*condition_columns)),
                    dtype=torch.long,
                )
                labels = [self._parse_multihot_sequence(row[field]) for field in self.label_fields]
                for label in labels:
                    if label.shape[0] != num_condition_rows:
                        raise ValueError(
                            "Label sequence length must match conditioning length for clip "
                            f"{clip_path}: {label.shape[0]} vs {num_condition_rows}"
                        )

                if clip_path in label_lookup:
                    raise ValueError(f"Duplicate multi-condition multilabel row for clip: {clip_path}")
                label_lookup[clip_path] = (condition_labels, labels)
        return label_lookup

    def __getattr__(self, name):
        try:
            base_dataset = object.__getattribute__(self, "base_dataset")
            return getattr(base_dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
