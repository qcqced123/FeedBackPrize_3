import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel
from ..base.base_model import BaseModel
from model_utils import init_weights, freeze, reinit_topk
from pooling import AttentionPooling, WeightedLayerPooling, MeanPooling


class FBPModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(BaseModel, self)
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model_name,
            output_hidden_states = True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config = self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, cfg.num_classes)

        """
        1) Set pooling method
        2) Reinit top-k encoder layers, fully connected layer
        3) Freeze bottom-k encoder layers
        4) Enable gradient checkpointing
        """
        if cfg.pooling == 'attention':
            self.pooling = AttentionPooling(self.auto_cfg.hidden_size)
        elif cfg.pooling == 'weighted':
            self.pooling = WeightedLayerPooling(self.auto_cfg.num_hidden_layers, cfg.layer_start, cfg.layer_weights)
        elif cfg.pooling == 'mean':
            self.pooling = MeanPooling()

        if cfg.reinit:
            init_weights(self.auto_cfg, self.fc)
            reinit_topk(self.backbone, cfg.num_reinit_layers)

        if cfg.freeze:
            freeze(self.backbone)

        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def feature(self, inputs: dict):
        outputs = self.backbone(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(**inputs)
        embedding = self.pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        logit = self.fc(embedding)
        return logit


class TeacherModel(BaseModel):
    """
    Teacher model for Meta Pseudo Label Pipeline
    """
    def __init__(self, cfg):
        super().__init__(BaseModel, self)
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model_name,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, cfg.num_classes)
        """
        1) Set pooling method
        2) Reinit top-k encoder layers, fully connected layer
        3) Freeze bottom-k encoder layers
        4) Enable gradient checkpointing
        """
        if cfg.pooling == 'attention':
            self.pooling = AttentionPooling(self.auto_cfg.hidden_size)
        elif cfg.pooling == 'weighted':
            self.pooling = WeightedLayerPooling(self.auto_cfg.num_hidden_layers, cfg.layer_start, cfg.layer_weights)
        elif cfg.pooling == 'mean':
            self.pooling = MeanPooling()

        if cfg.reinit:
            init_weights(self.auto_cfg, self.fc)
            reinit_topk(self.backbone, cfg.num_reinit_layers)

        if cfg.freeze:
            freeze(self.backbone)

        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def feature(self, inputs: dict):
        outputs = self.backbone(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(**inputs)
        embedding = self.pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        logit = self.fc(embedding)
        return logit


class StudentModel(BaseModel):
    """
    Student model for Meta Pseudo Label Pipeline
    """
    def __init__(self, cfg):
        super().__init__(BaseModel, self)
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model_name,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, cfg.num_classes)
        """
        1) Set pooling method
        2) Reinit top-k encoder layers, fully connected layer
        3) Freeze bottom-k encoder layers
        4) Enable gradient checkpointing
        """
        if cfg.pooling == 'attention':
            self.pooling = AttentionPooling(self.auto_cfg.hidden_size)
        elif cfg.pooling == 'weighted':
            self.pooling = WeightedLayerPooling(self.auto_cfg.num_hidden_layers, cfg.layer_start, cfg.layer_weights)
        elif cfg.pooling == 'mean':
            self.pooling = MeanPooling()

        if cfg.reinit:
            init_weights(self.auto_cfg, self.fc)
            reinit_topk(self.backbone, cfg.num_reinit_layers)

        if cfg.freeze:
            freeze(self.backbone)

        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def feature(self, inputs: dict):
        outputs = self.backbone(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(**inputs)
        embedding = self.pooling(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
        logit = self.fc(embedding)
        return logit
