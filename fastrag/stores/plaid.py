import logging
import math
from typing import List

import pandas as pd
from haystack.document_stores import BaseDocumentStore
from haystack.lazy_imports import LazyImport
from haystack.schema import Document
import torch
import copy

with LazyImport(
        "Run 'pip install libs/colbert' from root dir to install ColBERT lib"
) as colbert_import:
    from colbert import Indexer, Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
# from fastrag.stores.ColBERT.colbert import Indexer, Searcher
# from fastrag.stores.ColBERT.colbert.infra import ColBERTConfig, Run, RunConfig


logger = logging.getLogger(__name__)


class PLAIDDocumentStore(BaseDocumentStore):
    """
    Store for ColBERT v2 with PLAID indexing.

    Parameters:

    index_path: directory containing PLAID index files.
    checkpoint_path: directory containing ColBERT checkpoint model files.
    collection_path: a csv/tsv data file of the form (id,content), no header line.

    create: whether to create a new index or load an index from disk. Default: False.

    nbits: number of bits to quantize the residual vectors. Default: 2.
    kmeans_niters: number of kmeans clustering iterations. Default: 1.
    gpus: number of GPUs to use for indexing. Default: 0.
    rank: number of ranks to use for indexing. Default: 1.
    doc_maxlen: max document length. Default: 120.
    query_maxlen: max query length. Default: 60.

    """

    def __init__(
            self,
            index_path,
            checkpoint_path,
            collection_path,
            retrieval_config: dict,
            create=False,
            nbits=2,
            gpus=0,
            ranks=1,
            doc_maxlen=120,
            query_maxlen=60,
            kmeans_niters=4,
    ):
        colbert_import.check()
        super().__init__()
        self.index_path = index_path
        self.checkpoint_path = checkpoint_path
        self.collection_path = collection_path
        self.nbits = nbits
        self.gpus = gpus
        self.ranks = ranks
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.kmeans_niters = kmeans_niters

        if create:
            self._create_index()

        self.docs = pd.read_csv(
            collection_path, sep="\t" if collection_path.endswith(".tsv") else ",", header=None
        )
        self.titles = len(self.docs.columns) > 2
        self._load_index(retrieval_config=retrieval_config)

    def _load_index(self, retrieval_config: dict):
        """Load PLAID index from the paths given to the class and initialize a Searcher object."""
        if 'n_thread' in retrieval_config:
            torch.set_num_threads(retrieval_config['n_thread'])
        colbert_retrieval_config = copy.deepcopy(retrieval_config)
        if 'n_thread' in retrieval_config:
            del colbert_retrieval_config['n_thread']
        with Run().context(
                RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            config = ColBERTConfig(
                **colbert_retrieval_config
            )
            self.store = Searcher(
                index=self.index_path, collection=self.collection_path, checkpoint=self.checkpoint_path, config=config
            )

        logger.info("Loaded PLAIDDocumentStore index")

    def _create_index(self):
        """Generate a PLAID index from a given ColBERT checkpoint.

        Given a checkpoint and a collection of documents, an Indexer object will be created.
        The index will then be generated, written to disk at `index_path` and finally it
        will be loaded.
        """

        with Run().context(
                RunConfig(index_root=self.index_path, nranks=self.ranks, gpus=self.gpus)
        ):
            config = ColBERTConfig(
                doc_maxlen=self.doc_maxlen,
                query_maxlen=self.query_maxlen,
                nbits=self.nbits,
                kmeans_niters=self.kmeans_niters,
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(self.index_path, collection=self.collection_path, overwrite=True)

        logger.info("Created PLAIDDocumentStore Index.")

    def write_documents(self, dataset, batch_size=1):
        raise Exception(
            "PLAIDDocumentStore can only be used as a read-only store. A new index is needed for adding/changing documents"
        )

    def get_all_documents(self):
        raise NotImplementedError

    def get_all_documents_generator(self):
        raise NotImplementedError

    def delete_index(self):
        raise NotImplementedError

    def get_all_labels(self):
        raise NotImplementedError

    def query_by_embedding(self):
        raise NotImplementedError

    def get_label_count(self):
        raise NotImplementedError

    def write_labels(self):
        raise NotImplementedError

    def delete_documents(self):
        raise NotImplementedError

    def delete_labels(self):
        raise NotImplementedError

    def _create_document_field_map(self):
        raise NotImplementedError

    def get_documents_by_id(self):
        raise NotImplementedError

    def get_document_by_id(self):
        raise NotImplementedError

    def update_document_meta(self):
        raise NotImplementedError

    @staticmethod
    def _normalize_scores(docs: List[Document]) -> None:
        "Normalizing the MaxSim scores using softmax."
        Z = sum(math.exp(doc.score) for doc in docs)
        for doc in docs:
            doc.score = math.exp(doc.score) / Z

    def get_document_count(self):
        """
        Returns the number of docs in the collection.
        """
        return len(self.docs)

    def query(self, query_str, top_k=10) -> List[Document]:
        """
        Query the Colbert v2 + Plaid store.

        Returns: list of Haystack documents.
        """

        doc_ids, _, scores, ivf_time_ms, filter_time_ms, refine_time_ms, n_refine_ivf, n_refine_filter = self.store.search(
            text=query_str, k=top_k)

        documents = [
            Document.from_dict(
                {
                    "content": self.docs.iloc[_id][1],
                    "id": _id,
                    "score": score,
                    "meta": {"title": self.docs.iloc[_id][2] if self.titles else None},
                }
            )
            for _id, score in zip(doc_ids, scores)
        ]

        # self._normalize_scores(documents)

        return documents

    def query_batch(self, query_strs: List[str], top_k=10) -> List[List[Document]]:
        """
        Query batch the Colbert v2 + Plaid store.

        Returns: lists of lists of Haystack documents.
        """

        query = self.store.search_all({i: s for i, s in enumerate(query_strs)}, k=top_k)
        documents = []

        for result in query.data.values():
            s_docs = [
                Document.from_dict(
                    {
                        "content": self.docs.iloc[_id][1],
                        "id": _id,
                        "score": score,
                        "meta": {"title": self.docs.iloc[_id][2] if self.titles else None},
                    }
                )
                for _id, _, score in result
            ]
            documents.append(s_docs)

        # for docs in documents:
        #     self._normalize_scores(docs)

        return documents
