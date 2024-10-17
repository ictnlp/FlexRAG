# Librarian: An agent powered search engine.



## TODO
- [ ] optimize keyword searcher
- [ ] try to merge TextUnit/RetrievedContext/Passage
- [ ] Add new test cases & benchmark
- [ ] using httpx client optimize the API based methods(add proxy & retry support)
- [ ] Add Lance DB as a retriever
- [ ] Optimize the cache system




## Overview


## Installation

### Install from pip
```bash
pip install librarian
```


### Install from source
```bash
pip install pybind11

git clone https://tencent.zhangzhuocheng.top:3000/zhangzhuocheng/kylin
cd librarian
pip install ./
```

## Usage


## Tested HF Models

### Tested Encoders
- jina-embeddings-v3
- bge-m3
- contriever
- nomic-embed-text-v1.5
- msmarco-MiniLM-L-12-v3

### Tested ReRankers
- InRanker-base
- Jina-colbert-v2
- bge-reranker-v2-m3

### Tested Generators

