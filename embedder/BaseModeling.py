import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)

@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    
    
class BaseEmbedderModel(ABC, nn.Module):
    def __init__(
        self,
        base_model,
        tokenizer: AutoTokenizer = None,
        temperature: float = 1.0
    ):
        super(BaseEmbedderModel, self).__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        
        self.temperature = temperature
        
    @abstractmethod
    def encode(self, features):
        pass
    
    @abstractmethod
    def compute_loss(self, scores, target):
        pass
    
    @abstractmethod
    def compute_score(self, q_reps, p_reps):
        pass

    @abstractmethod
    def save(self, output_dir: str):
        pass
    
    def forward(
        self,
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,

    ):
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)
        scores, loss = self._compute_in_batch_neg_loss(q_reps, p_reps)
        
        # if self.training:
        #     scores, loss = self._compute_in_batch_neg_loss(q_reps, p_reps)
        # else:
        #     loss = None
        return EmbedderOutput(
            loss=loss,
        )
          
    def _compute_in_batch_neg_loss(self, q_reps, p_reps):
        """
        group_size = 8
        batch_size = 4
        idx = [0, 1, 2, 3]
        target = [0, 8, 16, 24] 代表每個query的正向passage的index
        
        """
        group_size = p_reps.size(0) // q_reps.size(0)
        scores = self.compute_score(q_reps, p_reps) # (batch_size, group_size)
        idx = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
        target = idx * group_size 
        loss = self.compute_loss(scores, target)
        
        return scores, loss