import logging
import os

import numpy as np

from index import FaissIndex


logging.basicConfig(level=logging.INFO)
DATA_PATH = "/data/zhangzhuocheng/Lab/Python/LLM/datasets/RAG/wikipedia/wiki_2021"


# open embeddings
print("Loading embeddings")
embeds = np.memmap(
    os.path.join(DATA_PATH, "contriever_embeddings", "embeds.npy"),
    mode="r",
    dtype=np.float16,
    shape=(33176581, 768),
)
print(embeds.shape)


# train IVFPQ indices
# index = FaissIndex(
#     index_path=os.path.join(DATA_PATH, "contriever_embeddings_4/IVFPQ.faiss"),
#     vector_sz=768,
#     n_subquantizers=64,
#     n_bits=8,
#     n_list=4096,
#     n_probe=36,
#     device_id=4,
#     train_num=1000000,
#     log_interval=100,
# )
# print("Training index")
# index.train_index(embeds)
# print("Adding index")
# index.add_embeddings(embeds, batch_size=1000)
# index.serialize()


# train PQ indices
# index = FaissIndex(
#     index_path=os.path.join(DATA_PATH, "contriever_embeddings_4/PQ.faiss"),
#     vector_sz=768,
#     n_subquantizers=64,
#     n_bits=8,
#     n_list=0,
#     n_probe=36,
#     device_id=4,
#     train_num=1000000,
#     log_interval=100,
# )
# print("Training PQ index")
# index.train_index(embeds)
# print("Adding PQ index")
# index.add_embeddings(embeds, batch_size=1000)
# index.serialize()


# train IVF indices
# index = FaissIndex(
#     index_path=os.path.join(DATA_PATH, "contriever_embeddings_4/IVF.faiss"),
#     vector_sz=768,
#     n_subquantizers=64,
#     n_bits=0,
#     n_list=4096,
#     n_probe=36,
#     device_id=-1,
#     train_num=1000000,
#     log_interval=100,
# )
# print("Training IVF index")
# index.train_index(embeds)
# print("Adding IVF index")
# index.add_embeddings(embeds, batch_size=1000)
# index.serialize()


# train HNSW indices
# index = FaissIndex(
#     index_path=os.path.join(DATA_PATH, "contriever_embeddings_4/HNSW.faiss"),
#     vector_sz=768,
#     n_subquantizers=64,
#     n_bits=0,
#     n_list=4096,
#     n_probe=36,
#     device_id=-1,
#     train_num=1000000,
#     log_interval=100,
# )
# print("Training IVF index")
# index.train_index(embeds)
# print("Adding IVF index")
# index.add_embeddings(embeds, batch_size=1000)
# index.serialize()
