from transformers import HfArgumentParser

from embedder import (
    BaseEmbeddingDataArguments,
    BaseEmbeddingTrainingArguments,
    BaseEmbeddingModelArguments,
)

from runner import EncoderOnlyEmbedderRunner

parser = HfArgumentParser((
    BaseEmbeddingTrainingArguments,
    BaseEmbeddingModelArguments,
    BaseEmbeddingDataArguments 
))

training_args, model_args, data_args = parser.parse_args_into_dataclasses()
model_args: BaseEmbeddingModelArguments
data_args: BaseEmbeddingDataArguments
training_args: BaseEmbeddingTrainingArguments


runner = EncoderOnlyEmbedderRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)

runner.run()