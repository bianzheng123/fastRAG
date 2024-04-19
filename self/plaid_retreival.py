from fastrag.stores import PLAIDDocumentStore
import fastrag, torch

username = 'bianzheng'
dataset = 'example'
index_path = f'/home/{username}/fastRAG/data/index/{dataset}'
collection_path = f'/home/{username}/fastRAG/data/collection/{dataset}-collection.tsv'

'''define retriever'''
store = PLAIDDocumentStore(index_path=index_path,
                           checkpoint_path="Intel/ColBERT-NQ",
                           collection_path=collection_path)

from fastrag.retrievers.colbert import ColBERTRetriever

retriever = ColBERTRetriever(store)
for i in range(10000):
    res = retriever.retrieve("What is Machine Learning?", 3)
    print(res)

'''define reader'''
# from fastrag.prompters.invocation_layers import fid
# from haystack.nodes import PromptModel
# from haystack.nodes.prompt.prompt_template import PromptTemplate
# from haystack.nodes.prompt import PromptNode
# import torch
#
# PrompterModel = PromptModel(
#     model_name_or_path="Intel/fid_flan_t5_base_nq",
#     use_gpu=True,
#     invocation_layer_class=fid.FiDHFLocalInvocationLayer,
#     model_kwargs=dict(
#         model_kwargs=dict(
#             device_map={"": 0},
#             torch_dtype=torch.bfloat16,
#             do_sample=False
#         ),
#         generation_kwargs=dict(
#             max_length=10
#         )
#     )
# )
#
# reader = PromptNode(
#     model_name_or_path=PrompterModel,
#     default_prompt_template=PromptTemplate("{query}")
# )
#
# '''define pipeline'''
# from haystack import Pipeline
#
# p = Pipeline()
# p.add_node(component=retriever, name="Retriever", inputs=["Query"])
# p.add_node(component=reader, name="Reader", inputs=["Retriever"])
# res = p.run(query="What did Einstein work on?")
# print(res.keys())
# print(res['results'][0])
