import argparse
import logging
from pathlib import Path

from fastrag.stores import PLAIDDocumentStore

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an index using PLAID engine as a backend")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--doc-max-length", type=int, default=120)
    parser.add_argument("--query-max-length", type=int, default=60)
    parser.add_argument("--kmeans-iterations", type=int, default=4)
    parser.add_argument("--name", type=str, default="plaid_index")
    parser.add_argument("--nbits", type=int, default=2)

    args = parser.parse_args()

    if args.gpus > 1:
        args.ranks = args.gpus
        args.amp = True
    assert args.ranks > 0
    if args.gpus == 0:
        assert args.ranks > 0

    username = args.username
    dataset = args.dataset

    checkpoint = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/colbert-pretrain/colbertv2.0'
    collection = f'/home/{username}/fastRAG/data/collection/{dataset}-collection.tsv'
    index_save_path = f'/home/{username}/fastRAG/data/index/{dataset}'

    store = PLAIDDocumentStore(
        index_path=f"{index_save_path}",
        checkpoint_path=f"{checkpoint}",
        collection_path=f"{collection}",
        retrieval_config={},
        create=True,
        nbits=args.nbits,
        gpus=args.gpus,
        ranks=args.ranks,
        doc_maxlen=args.doc_max_length,
        query_maxlen=args.query_max_length,
        kmeans_niters=args.kmeans_iterations,
    )
    logger.info("Done.")
