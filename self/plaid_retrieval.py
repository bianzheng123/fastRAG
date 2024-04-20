import time

from fastrag.stores import PLAIDDocumentStore
from fastrag.retrievers.colbert import ColBERTRetriever
from fastrag.prompters.invocation_layers import fid

from haystack.nodes import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes.prompt import PromptNode
import torch
import os
import numpy as np
import json

import performance_metric

from haystack import Pipeline


def load_pipeline(username: str, dataset: str, retrieval_config: dict):
    index_path = f'/home/{username}/fastRAG/data/index/{dataset}'
    collection_path = f'/home/{username}/fastRAG/data/collection/{dataset}-collection.tsv'

    '''define retriever'''
    store = PLAIDDocumentStore(index_path=index_path,
                               checkpoint_path="Intel/ColBERT-NQ",
                               collection_path=collection_path,
                               retrieval_config=retrieval_config,
                               doc_maxlen=180, query_maxlen=60, kmeans_niters=20)

    retriever = ColBERTRetriever(store)
    # for i in range(10000):
    #     res = retriever.retrieve("What is Machine Learning?", 3)
    #     print(res)

    '''define reader'''
    PrompterModel = PromptModel(
        model_name_or_path="Intel/fid_flan_t5_base_nq",
        use_gpu=True,
        invocation_layer_class=fid.FiDHFLocalInvocationLayer,
        model_kwargs=dict(
            model_kwargs=dict(
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                do_sample=False
            ),
            generation_kwargs=dict(
                max_length=10
            )
        )
    )

    reader = PromptNode(
        model_name_or_path=PrompterModel,
        default_prompt_template=PromptTemplate("{query}")
    )

    # '''define pipeline'''
    # p = Pipeline()
    # p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    # p.add_node(component=reader, name="Reader", inputs=["Retriever"])
    # res = p.run(query="What is Machine Learning?", params={"Retriever": {"top_k": topk}})
    return retriever, reader


def load_query_gnd(username: str, dataset: str):
    document_path = f'/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document'
    query_fname = os.path.join(document_path, 'queries.dev.tsv')
    query_l = []
    queryID_l = []
    with open(query_fname, 'r') as f:
        for line in f:
            query_id, query = line.strip().split('\t')
            queryID_l.append(int(query_id))
            query_l.append(query)

    queryID2gnd_m = performance_metric.read_exact_match_gnd(dataset=dataset, username=username)
    return queryID_l, query_l, queryID2gnd_m


def run_end2end(username: str, dataset: str, retrieval_parameter_l: list, topk_l: list, method_name: str):
    queryID_l, query_l, queryID2gnd_m = load_query_gnd(username=username, dataset=dataset)

    for topk in topk_l:
        for para in retrieval_parameter_l:
            retriever, reader = load_pipeline(username=username, dataset=dataset, retrieval_config=para)

            retrieval_time_l = []
            reading_time_l = []
            result_l = []
            for local_qID, query in enumerate(query_l):
                start_time = time.time()
                retriever_res = retriever.retrieve(query, topk)
                end_time = time.time()
                retrieval_time = end_time - start_time

                start_time = time.time()
                read_res = reader.run(query=query, documents=retriever_res, prompt_template=PromptTemplate("{query}"))
                end_time = time.time()
                reading_time = end_time - start_time

                retrieval_time_l.append(retrieval_time)
                reading_time_l.append(reading_time)
                results = read_res[0]['results']
                result_l.append(results)

            end2end_time_l = [retr + reading for retr, reading in zip(retrieval_time_l, reading_time_l)]

            search_time_m = {
                "end2end_time_p5(ms)": '{:.3f}'.format(np.percentile(end2end_time_l, 5) * 1e3),
                "end2end_time_p50(ms)": '{:.3f}'.format(np.percentile(end2end_time_l, 50) * 1e3),
                "end2end_time_p95(ms)": '{:.3f}'.format(np.percentile(end2end_time_l, 95) * 1e3),
                "end2end_time_average(ms)": '{:.3f}'.format(1.0 * np.average(end2end_time_l) * 1e3),

                "retrieval_time(ms)": '{:.3f}'.format(1.0 * np.average(retrieval_time_l) * 1e3),
                "reading_time(ms)": '{:.3f}'.format(1.0 * np.average(reading_time_l) * 1e3),
            }
            search_accuracy_m, exact_match_l = performance_metric.compute_exact_match(result_l=result_l,
                                                                                      queryID_l=queryID_l,
                                                                                      queryID2gnd_m=queryID2gnd_m)
            for local_qID, results, exact_match in zip(np.arange(len(result_l)), result_l, exact_match_l):
                print(f"local_qID {local_qID}, results {results}, exact_match {exact_match}")

            performance_m = {
                'n_query': len(query_l),
                'topk': topk,
                'retrieval': para,
                'search_time': search_time_m,
                'search_accuracy': search_accuracy_m
            }

            ndocs = para['ndocs']
            ncells = para['ncells']
            centroid_score_threshold = para['centroid_score_threshold']
            n_thread = para['n_thread']
            retrieval_suffix = f'ndocs_{ndocs}-ncells_{ncells}-centroid_score_threshold_{centroid_score_threshold}-n_thread_{n_thread}'
            method_performance_name = f'OpenQA-{dataset}-retrieval-{method_name}-top{topk}-{retrieval_suffix}.json'
            performance_dir = f'/home/{username}/Dataset/vector-set-similarity-search/Result/performance'
            performance_filename = os.path.join(performance_dir, method_performance_name)
            with open(performance_filename, "w") as f:
                json.dump(performance_m, f)
            print("#############final result###############")
            print("filename", performance_m['retrieval'])
            print("search time", performance_m['search_time'])
            print("search accuracy", performance_m['search_accuracy'])
            print("########################################")


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for ndocs in grid_search_para['ndocs']:
        for ncells in grid_search_para['ncells']:
            for centroid_score_threshold in grid_search_para['centroid_score_threshold']:
                for n_thread in grid_search_para['n_thread']:
                    parameter_l.append(
                        {"ndocs": ndocs, "ncells": ncells,
                         "centroid_score_threshold": centroid_score_threshold,
                         "n_thread": n_thread})
    return parameter_l


if __name__ == '__main__':
    # 'ndocs': searcher.config.ndocs,
    # 'ncells': searcher.config.ncells,
    # 'centroid_score_threshold': searcher.config.centroid_score_threshold
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            # 'dataset_l': ['lotte', 'msmacro'],
            # 'dataset_l': ['lotte-lifestyle', 'lotte', 'msmacro'],
            'dataset_l': ['wikipedia'],
            'topk_l': [10],
            'retrieval_parameter_l': [
                {"ndocs": 128, "ncells": 1, "centroid_score_threshold": 0.5, "n_thread": 1},
            ],
            'grid_search': True,
            'grid_search_para': {
                # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                'ndocs': [4 * 10, 4 * 50, 4 * 100, 4 * 200, 4 * 400, 4 * 800],
                'ncells': [1, 2],
                'centroid_score_threshold': [0.5],
                'n_thread': [1]
            }
        },
        'local': {
            'username': 'bianzheng',
            # 'dataset_l': ['lotte-500-gnd'],
            'dataset_l': ['wikipedia-500'],
            'topk_l': [10],
            'retrieval_parameter_l': [
                {'ndocs': 32, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 128, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 512, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1}
            ],
            'grid_search': False,
            'grid_search_para': {
                'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400],
                'ncells': [1, 2],
                'centroid_score_threshold': [0.5, 0.55],
                'n_thread': [1]
            }
        }
    }
    host_name = 'local'

    config = config_l[host_name]
    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']

    method_name = 'VSS-Colbert'

    grid_search = config['grid_search']
    if grid_search:
        retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'])
    else:
        retrieval_parameter_l = config['retrieval_parameter_l']

    for dataset in dataset_l:
        run_end2end(username=username, dataset=dataset, retrieval_parameter_l=retrieval_parameter_l, topk_l=topk_l,
                    method_name=method_name)
