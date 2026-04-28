from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


# 정책서 DB경로
# DATA_PATH = "./content/test"
# DB_PATH = "./vectorstores/db/"

# 결제 flow data DB경로
DATA_PATH = "./content/dependence"
DB_PATH = "./vectorstores/payment_dependence/"

# 결함유형 DB 경로
# DATA_PATH = "./content/defect_type"
# DB_PATH = "./vectorstores/defect_type_check_list_db/"

# 데이터 정제된 결함유형 DB 경로
# DATA_PATH = "./content/new_template"
# DB_PATH = "./vectorstores/D_type_search_db/"

loader = PyPDFDirectoryLoader(DATA_PATH)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
)
pages = text_splitter.split_documents(docs)

# 임베딩 처리(벡터 변환), 임베딩은 HuggingFaceEmbeddings 모델을 사용합니다.
# 아래 모델은 한글 임베딩을 잘 해준다고 해서 가져와 봤습니다. 계속 누군가 좋은 모델을 업데이트하니 수시로 확인해서 적용해 보는것을 추천합니다.
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-multitask",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
    # GPU Device 설정:
    # 저는 M1 맥북을 사용하기 때문에 mps로 설정합니다. 다른 서버역할을 하는 곳의 하드웨어에 맞추어서 설정을 변경해 주셔야 합니다.
    # RAG_Policy_Searcher.py의 embedding 설정도 동일하게 맞추어 주셔야 정상적으로 동작합니다. 싱크를 꼭 맞춰주셔요.
    # - NVidia GPU: "cuda"
    # - Mac M1, M2, M3: "mps"
    # - CPU: "cpu"
)


vectorstore = FAISS.from_documents(pages, embeddings)

vectorstore.save_local(DB_PATH)
