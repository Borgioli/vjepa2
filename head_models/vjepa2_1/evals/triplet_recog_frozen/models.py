# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SingleHeadMultiTaskClassifier(nn.Module):
    """
    Single head classifier that outputs predictions for multiple tasks
    using one pooled representation and one linear layer.
    
    This architecture forces the model to learn a unified representation
    for all tasks, rather than having separate query tokens and classifiers.
    """
    
    def __init__(
        self,
        num_classes_per_task: list,
        embed_dim: int,
        num_heads: int,
        depth: int,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_tasks = len(num_classes_per_task)
        self.num_classes_per_task = num_classes_per_task
        self.total_classes = sum(num_classes_per_task)
        
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        
        self.classifier = nn.Linear(embed_dim, self.total_classes, bias=True)
    
    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected at output of encoder")
            exit(1)
        
        x = self.pooler(x)  
        x = x.squeeze(1)
        
        logits = self.classifier(x)
        
        task_logits = []
        offset = 0
        for num_classes in self.num_classes_per_task:
            task_logits.append(logits[:, offset:offset + num_classes])
            offset += num_classes
        
        output = {}
        for i in range(self.num_tasks):
            output[f"task_{i}"] = task_logits[i]
        
        return output


class PooledFeatureMultiTaskClassifier(nn.Module):
    """
    Pool encoder token features into a single clip representation, then classify.

    Supports simple mean pooling or attention pooling over tokens, followed by
    either a linear classifier or an MLP classifier depending on whether hidden
    dimensions are provided.
    """

    def __init__(
        self,
        num_classes_per_task: list,
        embed_dim: int,
        feature_pool: str = "mean",
        num_heads: int = 16,
        depth: int = 1,
        classifier_hidden_dims=None,
        use_layer_norm: bool = True,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_tasks = len(num_classes_per_task)
        self.num_classes_per_task = num_classes_per_task
        self.total_classes = sum(num_classes_per_task)
        self.feature_pool = feature_pool
        self.input_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()

        if self.feature_pool not in ("mean", "attention"):
            raise ValueError(
                f"Unsupported feature_pool '{self.feature_pool}'. Expected one of: mean, attention"
            )

        if self.feature_pool == "attention":
            self.pooler = AttentivePooler(
                num_queries=1,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
                use_activation_checkpointing=use_activation_checkpointing,
            )
        else:
            self.pooler = None

        if classifier_hidden_dims is None:
            classifier_hidden_dims = []
        elif isinstance(classifier_hidden_dims, int):
            classifier_hidden_dims = [classifier_hidden_dims]
        else:
            classifier_hidden_dims = [int(hidden_dim) for hidden_dim in classifier_hidden_dims]

        dims = [embed_dim] + classifier_hidden_dims + [self.total_classes]
        layers = []
        for layer_idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1], bias=True))
            if layer_idx < len(dims) - 2:
                layers.append(nn.GELU())
        self.classifier = nn.Sequential(*layers)

    def _pool_features(self, x):
        if self.feature_pool == "mean":
            return x.mean(dim=1)
        if self.feature_pool == "attention":
            return self.pooler(x).squeeze(1)
        raise RuntimeError(f"Unexpected feature_pool '{self.feature_pool}'")

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected at output of encoder")
            exit(1)

        x = self.input_norm(x)
        pooled = self._pool_features(x)
        logits = self.classifier(pooled)

        task_logits = []
        offset = 0
        for num_classes in self.num_classes_per_task:
            task_logits.append(logits[:, offset:offset + num_classes])
            offset += num_classes

        output = {}
        for i in range(self.num_tasks):
            output[f"task_{i}"] = task_logits[i]

        return output


class TokenAggregationMultiTaskClassifier(nn.Module):
    """
    Token-wise classifier that preserves encoder tokens until after classification.

    Each encoder token produces per-task logits first, then logits are aggregated
    across the token dimension into clip-level predictions.
    """

    def __init__(
        self,
        num_classes_per_task: list,
        embed_dim: int,
        token_pool: str = "max",
        token_pool_topk: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_tasks = len(num_classes_per_task)
        self.num_classes_per_task = num_classes_per_task
        self.total_classes = sum(num_classes_per_task)
        self.token_pool = token_pool
        self.token_pool_topk = token_pool_topk
        self.input_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.classifier = nn.Linear(embed_dim, self.total_classes, bias=True)

        if self.token_pool not in ("max", "mean", "logsumexp", "topk_mean"):
            raise ValueError(
                f"Unsupported token_pool '{self.token_pool}'. Expected one of: max, mean, logsumexp, topk_mean"
            )
        if self.token_pool == "topk_mean" and self.token_pool_topk is not None and self.token_pool_topk < 1:
            raise ValueError("token_pool_topk must be >= 1 when using token_pool='topk_mean'")

    def _aggregate_tokens(self, token_logits):
        token_dim = token_logits.ndim - 2
        if self.token_pool == "max":
            return token_logits.max(dim=token_dim).values
        if self.token_pool == "mean":
            return token_logits.mean(dim=token_dim)
        if self.token_pool == "logsumexp":
            return torch.logsumexp(token_logits, dim=token_dim)
        if self.token_pool == "topk_mean":
            k = token_logits.shape[token_dim] if self.token_pool_topk is None else min(
                self.token_pool_topk,
                token_logits.shape[token_dim],
            )
            return torch.topk(token_logits, k=k, dim=token_dim).values.mean(dim=token_dim)
        raise RuntimeError(f"Unexpected token_pool '{self.token_pool}'")

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected at output of encoder")
            exit(1)

        x = self.input_norm(x)
        token_logits = self.classifier(x)
        logits = self._aggregate_tokens(token_logits)

        task_logits = []
        offset = 0
        for num_classes in self.num_classes_per_task:
            task_logits.append(logits[:, offset:offset + num_classes])
            offset += num_classes

        output = {}
        for i in range(self.num_tasks):
            output[f"task_{i}"] = task_logits[i]

        return output


class TokenAggregationClassifier(nn.Module):
    """
    Single-task token-wise classifier that aggregates token logits into a
    clip-level prediction.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        token_pool: str = "max",
        token_pool_topk: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.token_pool = token_pool
        self.token_pool_topk = token_pool_topk
        self.input_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.classifier = nn.Linear(embed_dim, self.num_classes, bias=True)

        if self.token_pool not in ("max", "mean", "logsumexp", "topk_mean"):
            raise ValueError(
                f"Unsupported token_pool '{self.token_pool}'. Expected one of: max, mean, logsumexp, topk_mean"
            )
        if self.token_pool == "topk_mean" and self.token_pool_topk is not None and self.token_pool_topk < 1:
            raise ValueError("token_pool_topk must be >= 1 when using token_pool='topk_mean'")

    def _aggregate_tokens(self, token_logits):
        token_dim = token_logits.ndim - 2
        if self.token_pool == "max":
            return token_logits.max(dim=token_dim).values
        if self.token_pool == "mean":
            return token_logits.mean(dim=token_dim)
        if self.token_pool == "logsumexp":
            return torch.logsumexp(token_logits, dim=token_dim)
        if self.token_pool == "topk_mean":
            k = token_logits.shape[token_dim] if self.token_pool_topk is None else min(
                self.token_pool_topk,
                token_logits.shape[token_dim],
            )
            return torch.topk(token_logits, k=k, dim=token_dim).values.mean(dim=token_dim)
        raise RuntimeError(f"Unexpected token_pool '{self.token_pool}'")

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected at output of encoder")
            exit(1)

        x = self.input_norm(x)
        token_logits = self.classifier(x)
        return self._aggregate_tokens(token_logits)


class ConditionedTokenAggregationMultiTaskClassifier(nn.Module):
    """
    Token-wise multi-task classifier conditioned on one or more discrete
    auxiliary labels.

    The architecture mirrors TokenAggregationMultiTaskClassifier, but adds an
    embedding of the conditioning label(s) to each encoder token before
    classification.
    """

    expects_conditioning = True

    def __init__(
        self,
        num_classes_per_task: list,
        embed_dim: int,
        num_condition_classes,
        condition_embed_dim: int = None,
        token_pool: str = "max",
        token_pool_topk: int = None,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_tasks = len(num_classes_per_task)
        self.num_classes_per_task = num_classes_per_task
        self.total_classes = sum(num_classes_per_task)
        self.token_pool = token_pool
        self.token_pool_topk = token_pool_topk
        self.input_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()

        if isinstance(num_condition_classes, int):
            condition_class_counts = [num_condition_classes]
        else:
            condition_class_counts = [int(class_count) for class_count in num_condition_classes]
        if len(condition_class_counts) == 0:
            raise ValueError("num_condition_classes must contain at least one class count")

        self.num_condition_inputs = len(condition_class_counts)
        if condition_embed_dim is None:
            condition_embed_dims = [embed_dim] * self.num_condition_inputs
        elif isinstance(condition_embed_dim, int):
            condition_embed_dims = [condition_embed_dim] * self.num_condition_inputs
        else:
            condition_embed_dims = [int(dim) for dim in condition_embed_dim]
            if len(condition_embed_dims) != self.num_condition_inputs:
                raise ValueError(
                    "condition_embed_dim must be either an int or a list matching num_condition_classes"
                )

        self.condition_embeddings = nn.ModuleList(
            [
                nn.Embedding(class_count, embed_dim_i)
                for class_count, embed_dim_i in zip(condition_class_counts, condition_embed_dims)
            ]
        )
        self.condition_projections = nn.ModuleList(
            [
                nn.Identity() if embed_dim_i == embed_dim else nn.Linear(embed_dim_i, embed_dim)
                for embed_dim_i in condition_embed_dims
            ]
        )
        self.classifier = nn.Linear(embed_dim, self.total_classes, bias=True)

        if self.token_pool not in ("max", "mean", "logsumexp", "topk_mean"):
            raise ValueError(
                f"Unsupported token_pool '{self.token_pool}'. Expected one of: max, mean, logsumexp, topk_mean"
            )
        if self.token_pool == "topk_mean" and self.token_pool_topk is not None and self.token_pool_topk < 1:
            raise ValueError("token_pool_topk must be >= 1 when using token_pool='topk_mean'")

    def _aggregate_tokens(self, token_logits):
        token_dim = token_logits.ndim - 2
        if self.token_pool == "max":
            return token_logits.max(dim=token_dim).values
        if self.token_pool == "mean":
            return token_logits.mean(dim=token_dim)
        if self.token_pool == "logsumexp":
            return torch.logsumexp(token_logits, dim=token_dim)
        if self.token_pool == "topk_mean":
            k = token_logits.shape[token_dim] if self.token_pool_topk is None else min(
                self.token_pool_topk,
                token_logits.shape[token_dim],
            )
            return torch.topk(token_logits, k=k, dim=token_dim).values.mean(dim=token_dim)
        raise RuntimeError(f"Unexpected token_pool '{self.token_pool}'")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.num_condition_inputs == 1:
            old_embedding_key = f"{prefix}condition_embedding.weight"
            new_embedding_key = f"{prefix}condition_embeddings.0.weight"
            if old_embedding_key in state_dict and new_embedding_key not in state_dict:
                state_dict[new_embedding_key] = state_dict.pop(old_embedding_key)

            old_projection_weight_key = f"{prefix}condition_projection.weight"
            old_projection_bias_key = f"{prefix}condition_projection.bias"
            new_projection_weight_key = f"{prefix}condition_projections.0.weight"
            new_projection_bias_key = f"{prefix}condition_projections.0.bias"
            if old_projection_weight_key in state_dict and new_projection_weight_key not in state_dict:
                state_dict[new_projection_weight_key] = state_dict.pop(old_projection_weight_key)
            if old_projection_bias_key in state_dict and new_projection_bias_key not in state_dict:
                state_dict[new_projection_bias_key] = state_dict.pop(old_projection_bias_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, condition_labels):
        if torch.isnan(x).any():
            print("NaN detected at output of encoder")
            exit(1)
        if condition_labels is None:
            raise ValueError("ConditionedTokenAggregationMultiTaskClassifier requires condition_labels")

        x = self.input_norm(x)
        condition_labels = condition_labels.to(device=x.device, dtype=torch.long)
        if condition_labels.ndim == 1:
            condition_labels = condition_labels.unsqueeze(1)
        multi_condition = condition_labels.ndim == 3
        condition_input_dim = 2 if multi_condition else 1
        if condition_labels.shape[condition_input_dim] != self.num_condition_inputs:
            raise ValueError(
                f"Expected {self.num_condition_inputs} conditioning inputs, "
                f"got tensor with shape {tuple(condition_labels.shape)}"
            )

        condition_embedding = None
        for input_idx, (embedding_layer, projection_layer) in enumerate(
            zip(self.condition_embeddings, self.condition_projections)
        ):
            if multi_condition:
                embedded = projection_layer(embedding_layer(condition_labels[:, :, input_idx]))
            else:
                embedded = projection_layer(embedding_layer(condition_labels[:, input_idx]))
            condition_embedding = embedded if condition_embedding is None else condition_embedding + embedded

        if multi_condition and self.token_pool == "mean":
            # For the linear classifier and mean token pooling, this is exactly
            # equivalent to classifying x + condition_embedding for every pair,
            # without materializing [batch, pairs, tokens, dim].
            base_logits = self.classifier(x).mean(dim=1)
            condition_delta = torch.matmul(condition_embedding, self.classifier.weight.t())
            logits = base_logits.unsqueeze(1) + condition_delta
        else:
            if multi_condition:
                x = x.unsqueeze(1) + condition_embedding.unsqueeze(2)
            else:
                x = x + condition_embedding.unsqueeze(1)

            token_logits = self.classifier(x)
            logits = self._aggregate_tokens(token_logits)

        task_logits = []
        offset = 0
        for num_classes in self.num_classes_per_task:
            task_logits.append(logits[..., offset:offset + num_classes])
            offset += num_classes

        output = {}
        for i in range(self.num_tasks):
            output[f"task_{i}"] = task_logits[i]

        return output


def init_module(
    module_name,
    device,
    frames_per_clip,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for Encoder module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model
