import logging as logger

from haystack.preprocessor import PreProcessor
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.reader.farm import FARMReader

logger.basicConfig(level="INFO")


document_store = ElasticsearchDocumentStore(host="elasticsearch", username="", password="", index="document-demo")

processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=150,
    split_respect_sentence_boundary=True,
    split_overlap=0
)

logger.info("Initialization Of DPR.")

retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)

logger.info("Initialization of reader.")

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

logger.info("Building pipeline.")
pipe = ExtractiveQAPipeline(reader, retriever)