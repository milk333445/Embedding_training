import logging

import torch
from transformers import AutoModel, AutoTokenizer


from embedder import BaseEmbedderModel

logger = logging.getLogger(__name__)


class BiEncoderOnlyEmbedderModel(BaseEmbedderModel):
    TRANSFORMER_CLS = AutoModel
    
    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        temperature: float = 1.0,
        sentence_pooling_method: str = 'cls',
        normalize_embedding: bool = False,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            temperature=temperature
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embedding = normalize_embedding
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        
    def encode(self, features):
        if features is None:
            return None
        