import typing
import pydantic

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from twon_metrics.hf_classify import HFClassify
from twon_metrics.api.config import Config

DEVICE: torch.device = torch.device("cuda:3")


MODELS: typing.Tuple[str, str] = [
    ("topics", "cardiffnlp/tweet-topic-21-multi"),
    ("emotions", "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"),
    ("sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("irony", "cardiffnlp/twitter-roberta-base-irony"),
    ("offensive",  "cardiffnlp/twitter-roberta-base-offensive"),
    ("hate", "cardiffnlp/twitter-roberta-base-hate-latest")
]

classfifiers: typing.Dict[str, HFClassify] = {
    label: HFClassify(slug, device=DEVICE)
    for label, slug in MODELS
}


class Request(pydantic.BaseModel):
    samples: typing.List[str]
    threshold: float = 0.5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        "Alright guys I'm falling sleeping sitting up with my phone in my hand.. that would be my clue. Goodnight Twitter family!",
                        "Outrageous! A new report exposes the stark wealth inequality in corporate America. Over 50 major companies have CEOs earning more than thousands of times their median worker's salary! This is not just a moral issue, it's an economic one.",
                        "If Biden were serious about fixing this border crisis, he should have announced he was reinstating our policies that worked. Visiting isn't enough.",
                        "Great to see the job market bouncing back! But let's not forget that this growth is largely driven by low-wage, precarious work. We need policies that prioritize good jobs, affordable healthcare, and a living wage for all, not just corporate profits. #workersrights #economicjustice"
                    ],
                    "threshold": 0.5
                }
            ]
        }
    }


class Prediction(pydantic.BaseModel):
    sample: str
    results: typing.Dict[str, typing.Dict[str, float]]


class Response(pydantic.BaseModel):
    predictions: typing.List[Prediction]

cfg = Config()

app = FastAPI(
    title=cfg.title,
    version=cfg.version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.trust_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@app.post("/")
async def calc_metric(req: Request) -> Response:

    collated_logits: typing.Dict = {
        label: [] for label in classfifiers.keys()
    }

    for batch in batched(req.samples, 64):
        for label, classfifier in classfifiers.items():
           collated_logits[label].extend(classfifier(batch, theta=req.threshold))

    return Response(
        predictions=[
            Prediction(
                sample=sample,
                results={
                    label: preds[n]
                    for label, preds in collated_logits.items()
                }
            )
            for n, sample in enumerate(req.samples)
        ]
    )


__all__ = ["Config", "app"]