import uvicorn

from twon_metrics.api import app


if __name__ == "__main__":
    uvicorn.run(app)