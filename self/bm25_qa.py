from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)

from haystack.schema import Document

# 3 example documents to index
examples = [
    "There is a blue house on Oxford street",
    "Paris is the capital of France",
    "fastRAG had its first commit in 2022"
]

documents = []
for i, d in enumerate(examples):
    documents.append(Document(content=d, id=i))

document_store.write_documents(documents)

from haystack.nodes import BM25Retriever

# define a BM25 retriever, ST re-ranker and FiD reader based on a local model
retriever = BM25Retriever(document_store=document_store)

from fastrag.prompters.invocation_layers import fid
from haystack.nodes import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.nodes.prompt import PromptNode
import torch

PrompterModel = PromptModel(
    model_name_or_path="Intel/fid_flan_t5_base_nq",
    # model_name_or_path="google/flan-t5-base",
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

# from haystack.nodes import TransformersReader
# reader = TransformersReader(model_name_or_path="distilbert/distilbert-base-uncased-distilled-squad")

from haystack import Pipeline

p = Pipeline()

p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reader, name="Reader", inputs=["Retriever"])

res = p.run(query="What is Paris?")
print(res.keys())
print(res['results'])
