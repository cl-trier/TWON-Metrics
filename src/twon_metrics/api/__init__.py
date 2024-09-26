import typing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from twon_metrics.hf_classify import HFClassify
from twon_metrics.api import schemas, util
from twon_metrics.api.config import Config

CFG: Config = Config()


classfifiers: typing.Dict[str, HFClassify] = {
    label: HFClassify(slug, device=CFG.device) for label, slug in CFG.models
}

app = FastAPI(title=CFG.title, version=CFG.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.trust_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def calc_metric(req: schemas.Request) -> schemas.Response:
    collated_logits: typing.Dict[str, typing.List] = {
        label: [] for label in classfifiers.keys()
    }

    for batch in util.batched(req.samples, CFG.batch_size):
        for label, classfifier in classfifiers.items():
            collated_logits[label].extend(classfifier(batch, theta=req.threshold))

    return schemas.Response(
        predictions=[
            schemas.Prediction(
                sample=sample,
                results={label: preds[n] for label, preds in collated_logits.items()},
            )
            for n, sample in enumerate(req.samples)
        ]
    )


__all__ = ["Config", "app"]
