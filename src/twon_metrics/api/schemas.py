import typing
import pydantic


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