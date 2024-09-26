import typing
import pathlib

import torch
import pydantic


class Config(pydantic.BaseModel):
    title: str = "TWON Metrics API"
    version: str = "0.1.0"

    trust_origins: typing.List[str] = ["*"]

    models: typing.Tuple[str, str] = [
        ("topics", "cardiffnlp/tweet-topic-21-multi"),
        ("emotions", "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"),
        ("sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
        ("irony", "cardiffnlp/twitter-roberta-base-irony"),
        ("offensive", "cardiffnlp/twitter-roberta-base-offensive"),
        ("hate", "cardiffnlp/twitter-roberta-base-hate-latest"),
    ]

    device: torch.device = torch.device("cuda:3")
    batch_size: int = 256

    log_path: str = ".logs/"

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: typing.Any) -> None:
        self.log_path = f"{self.log_path}/{self.version}"
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
