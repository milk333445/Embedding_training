import logging
from typing import Tuple
from transformers import (
    AutoModel,AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from embedder import BaseEmbedderRunner, BaseEmbedderModel
from modeling import BiEncoderOnlyEmbedderModel
from trainer import EncoderOnlyEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderRunner(BaseEmbedderRunner):
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, BaseEmbedderModel]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        base_model = AutoModel.from_pretrained(self.model_args.model_name_or_path)
        
        model = BiEncoderOnlyEmbedderModel(
            base_model,
            tokenizer,
            temperature=self.training_args.temperature,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embedding=self.training_args.normalize_embeddings
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            
        return tokenizer, model
    
    def load_trainer(self) -> EncoderOnlyEmbedderTrainer:
        trainer = EncoderOnlyEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        return trainer
            
        