import os
import math
import random
import logging
import datasets
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from .BaseArguments import BaseEmbeddingDataArguments, BaseEmbeddingTrainingArguments


logger = logging.getLogger(__name__)

class BaseEmbedderTrainDataset(Dataset):
    def __init__(
        self, 
        args: BaseEmbeddingDataArguments,
        tokenizer: PreTrainedTokenizer
        ):
        self.args = args
        self.tokenizer = tokenizer
        
        train_datasets = []
        if os.path.isdir(s=args.train_data):
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset(
                    path='json',
                    data_files=os.path.join(args.train_data, file),
                    split='train',
                    cache_dir=args.cache_path
                )
                train_datasets.append(temp_dataset)
            self.train_dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.train_dataset = datasets.load_dataset(
                path='json',
                data_files=args.train_data,
                split='train',
                cache_dir=args.cache_path
            )
            
    def __len__(self):
        return len(self.train_dataset)
    
    def __getitem__(self, item):
        data = self.train_dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        passages = []
        
        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)
        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(data['pos'][pos_idx])
        
        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])
        
        return query, passages
    
@dataclass
class BaseEmbedderCollator(DataCollatorWithPadding):
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        queries_inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        q_collated = self.tokenizer.pad(
            queries_inputs,
            padding=self.padding,
            max_length=self.query_max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        d_collated = self.tokenizer.pad(
            passages_inputs,
            padding=self.padding,
            max_length=self.passage_max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )

        return {
            "queries": q_collated,
            "passages": d_collated
        }
    
    
    
    
    
if __name__ == '__main__':
    dataargument = BaseEmbeddingDataArguments(
        train_data='../../../dataset',
        cache_path='cache'
    )
    dataset=  BaseEmbedderTrainDataset(dataargument)
    print(len(dataset))
    query, passages = dataset[0]
    print(len(passages))
    
    datacollator = BaseEmbedderCollator()
    