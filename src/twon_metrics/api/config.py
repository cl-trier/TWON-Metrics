import typing
import pathlib


class Config:
    title: str = "TWON Metrics API"
    version: str = "0.1.0"

    trust_origins: typing.List[str] = ['*']

    log_path: str = ".logs/"

    def __init__(self) -> None:
        self.log_path = f"{self.log_path}/{self.version}"
        pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)