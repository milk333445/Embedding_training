import os
from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments

@dataclass
class BaseEmbeddingModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint and tokenizer and config"}
    )
    
@dataclass
class BaseEmbeddingDataArguments:
    train_data: str = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the cache directory."},
    )
    
    train_group_size: int = field(default=8)
    
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set will pad the sequence to be a multiple of the provided value."
        },
    )
    
    query_max_len: int = field(
        default=32,
        metadata={"help": "the maximum input length"}
    )
    
    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated."
        },
    )
    
@dataclass
class BaseEmbeddingTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=0.02)
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method. Available options: cls"})
    normalize_embeddings: bool = field(default=True, metadata={"help": "whether to normalize the embeddings"})
    sub_batch_size: Optional[int] = field(default=None, metadata={"help": "sub batch size for training"})
