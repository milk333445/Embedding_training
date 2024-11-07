import logging
from typing import Optional
from abc import ABC, abstractmethod
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)

class BaseEmbedderTrainer(ABC, Trainer):
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss