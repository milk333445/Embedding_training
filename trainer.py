import os
import torch
import logging
from typing import Dict, List, Union, Optional

from embedder import BaseEmbedderTrainer

logger = logging.getLogger(__name__)

class EncoderOnlyEmbedderTrainer(BaseEmbedderTrainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")
        self.model.save(output_dir)
        self.tokenizer.save_pretrained(output_dir)