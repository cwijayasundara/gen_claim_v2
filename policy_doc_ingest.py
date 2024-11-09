import warnings
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

vector_db_loc = "./policy_kb"

vector_db_name = "policy_docs"

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="This is an auto insurance claim documents and insurence policy document that contains information about claims and cashback limits",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    show_progress=True,
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(input_files=['docs/pb116349-business-health-select-handbook-1024-pdfa.pdf',
                                               'claim_forms/claim_1.pdf',
                                               'claim_forms/claim_2.pdf',
                                               'claim_forms/claim_3.pdf',
                                               'claim_forms/claim_4.pdf',
                                               'claim_forms/claim_5.pdf',],                      
                                  file_extractor=file_extractor).load_data()

db = chromadb.PersistentClient(path=vector_db_loc)

chroma_collection = db.get_or_create_collection(vector_db_name)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Test : Query Data from the persisted index
# query_1 = "Given the accident that happened on Lombard Street, name a party that is liable for the damages and explain why?"
# query_engine = index.as_query_engine()
# response_1 = query_engine.query(query_1) 
# print(response_1)