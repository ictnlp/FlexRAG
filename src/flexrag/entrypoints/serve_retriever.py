from dataclasses import asdict

import hydra
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from flexrag.retriever import FlexRetriever, FlexRetrieverConfig
from flexrag.utils import LOGGER_MANAGER, configure, extract_config

app = FastAPI()


retriever: FlexRetriever


@configure
class Config(FlexRetrieverConfig):
    host: str = "0.0.0.0"
    port: int = 3402
    read_only: bool = True


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("serve_retriever")


class SearchRequest(BaseModel):
    queries: list[str] = Field(
        description="List of queries to search for",
    )
    top_k: int = Field(
        default=10,
        description="Number of top results to return",
    )
    in_batches: bool = Field(
        default=False,
        description="Whether to process queries in batches",
    )
    batch_size: int = Field(
        default=32,
        description="Size of each batch if in_batches is true",
    )


@app.post("/search")
async def search(args: SearchRequest):
    try:
        if args.in_batches:
            responses = retriever.search_batch(
                query=args.queries,
                top_k=args.top_k,
                batch_size=args.batch_size,
                no_preprocess=True,
            )
        else:
            responses = retriever.search(
                query=args.queries,
                top_k=args.top_k,
            )
        responses = [[asdict(r) for r in response] for response in responses]
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    cfg = extract_config(cfg, Config)
    global retriever
    retriever = FlexRetriever(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
