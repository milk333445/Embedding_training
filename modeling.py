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
        if not isinstance(features, list):
            last_hidden_states = self.model(**features).last_hidden_state
            all_p_reps = self._sentence_embedding(last_hidden_states)
            if self.normalize_embedding:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()
        
    def _sentence_embedding(self, hidden_state):
        if self.sentence_pooling_method == "cls":
            return hidden_state[:, 0, :]
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")
        
    def _compute_similarity(self, q_reps, p_reps):
        if len(q_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1)) 
        return torch.matmul(q_reps, p_reps.transpose(-2, -1)) 
    
    def compute_score(self, q_reps, p_reps):
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores
        
    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


        