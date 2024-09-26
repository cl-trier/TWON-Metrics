import typing

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFClassify:
    # https://huggingface.co/cardiffnlp/tweet-topic-21-multi
    # https://arxiv.org/abs/2209.09824

    def __init__(self, slug: str, device: torch.device = torch.device("cpu")):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(slug)
        self.model = AutoModelForSequenceClassification.from_pretrained(slug).to(device)

        self.normalize_fn = torch.nn.Sigmoid()

    def __call__(self, batch: typing.List[str], theta: float = 0.25) -> pd.Series:
        return self.extract_label(self.model_forward(batch), theta)

    @torch.no_grad()
    def model_forward(self, batch: typing.List[str]) -> torch.tensor:
        return self.model(
            **self.tokenizer.batch_encode_plus(
                batch,
                truncation=True,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
            ).to(self.device)
        ).logits

    def extract_label(
        self, batch_logits: torch.tensor, theta: int
    ) -> typing.Iterator[typing.Dict[str, float]]:
        batch_norm_logits: torch.tensor = self.normalize_fn(batch_logits)
        batch_ids: torch.tensor = [
            preds.nonzero().squeeze().tolist()
            for preds in torch.unbind((batch_norm_logits >= theta).int())
        ]

        for n, post_ids in enumerate(batch_ids):
            if isinstance(post_ids, list):
                yield {
                    self.model.config.id2label[i]: batch_norm_logits[n, i].item()
                    for i in post_ids
                }

            else:
                yield {
                    self.model.config.id2label[post_ids]: batch_norm_logits[
                        n, post_ids
                    ].item()
                }
