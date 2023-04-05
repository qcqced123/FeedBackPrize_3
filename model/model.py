import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import init_weights, freeze, reinit_topk


class FBPModel(nn.Module):
    """ Model class for Baseline Pipeline """
    def __init__(self, cfg):
        super().__init__()
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.backbone,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.backbone,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 6)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)

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


class TeacherModel(nn.Module):
    """
    Teacher model for Meta Pseudo Label Pipeline
    """
    def __init__(self, cfg):
        super().__init__()
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model_name,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, cfg.num_classes)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)

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


class StudentModel(nn.Module):
    """
    Student model for Meta Pseudo Label Pipeline
    """
    def __init__(self, cfg):
        super().__init__()
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model_name,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, cfg.num_classes)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)

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
