import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

from operator import itemgetter


st.set_page_config(page_title="AI챗봇", page_icon="./onestore.png")
st.title("SDK 구매플로우 분석 💬")

# streamlit은 계속 처음부터 다시 그리는 특징이 있습니다. 그래서 대화상태를 저장해서 계속 뿌려주는 코드가 필요합니다.
# 채팅 히스토리가 없다면 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# 기존 대화내역 출력
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 저장된 채팅 히스토리가 없다면 초기화
if "store" not in st.session_state:
    st.session_state["store"] = dict()


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 세션 ID를 기반으로 세션 기록을 가져오는 함수, 해당함수를 통해 기존 대화내용 + 질문을 가지고 LLM이 응답을 해줍니다.
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    # print(session_ids)
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embedfile(file):
    file_content = file.read()
    file_path = f"./tempp/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=50,
    )
    pages = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    vectorstore = FAISS.from_documents(pages, embeddings)
    # 저장된 데이터의 검색기를 선언합니다.
    return vectorstore.as_retriever()


# 답변이 한번에 찍어주는데, 짧은 단어단위로 실시간 스트리밍 하게 하는 객체
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# 여기서부터 챗봇 UI입니다. 위에는 채팅기능을 위한 세부기능, RAG를 위한 데이터 전처리로 봐주시면 됩니다.
with st.sidebar:

    if st.button("대화기록 초기화"):
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.experimental_rerun()

    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

if uploaded_file:
    retriever = embedfile(uploaded_file)


# 채팅 입력
if user_input := st.chat_input("무엇이 궁금하신가요?"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    if uploaded_file:

        # 채팅 답변
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            # Stream handler를 설정하지 않으면 문장을 모두 완성한 후에 한번에 입력됩니다. 그래서 사용자가 딜레이를 더 크게 느끼게 됩니다.
            # 설정한다면, 음절단위로 출력하게끔 되어서 좀 더 일찍 답변을 받아볼 수 있게 됩니다.

            # LLM 모델설정을 설정합니다. ollama 의 로컬 LLM을 사용하기 때문에 미리 백그라운드에 설정이 필요합니다.
            # 이번에 사용한 모델은 gemma4 26b 모델입니다.
            llm = ChatOllama(
                # model="exaone3.5:latest",
                model="gemma4:26b-a4b-it-q8_0",
                streaming=True,
                callbacks=[stream_handler],
            )

            # 아래는 프롬프트로 이것만 수정해도 LLM 답변의 질이 높아 질 수 있습니다.
            prompt = PromptTemplate.from_template(
                """너는 컨텍스트를 기반으로 해서 질문에 대한 답변을 하는 챗봇이야. 
                Use the following pieces of retrieved context to answer the question. 
                주어진 컨텍스트는 일련의 프로세스를 정리한 문서야.
                그 프로세스 중에 일부분을 수정할 경우에 프로세스에 어떤영향이 있을지 분석이 필요해.
                답변의 길이가 길 경우에는, 가독성이 좋게 정리를 부탁해.
                If you don't know the answer, just say that you don't know. 
                Answer in Korean.

                #Previous Chat History:
                {history}

                #Question: 
                {question} 

                #Context: 
                {context} 

                #Answer:"""
            )

            # 체인을 생성합니다.
            # context는 참조할 데이터, 여기에서는 정책서가 해당됩니다. 질문이 리트리버(검색기)로 들어가서 관련내용을 찾고 format_docs로 파편화된 문서를 하나로 합쳐줍니다.
            # question은 사용자의 질문입니다.
            # history는 이전의 대화기록입니다.
            # | 이 파이프라인 명령어를 통해 위의 데이터가 프롬프트의 {} 변수자리에 들어갑니다. 그리고 그 결과가 llm으로 가서 답변을 생성, 그리고 stroutputparser로 사용자에게 답변할 내용을 파싱합니다.
            chain = (
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                    "history": itemgetter("history"),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # 이전 대화기록을 답변에 참고 할 수 있게 하는 코드
            # key를 아래처럼 선언함으로써 사용자의 질문과 이전 대화내용을 특정해서 이후 chain에 전달할 수 있습니다.
            chain_with_memory = (
                RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                    chain,  # 실행할 chain 객체
                    get_session_history,  # 세션 기록을 가져오는 함수
                    input_messages_key="question",  # 입력 메시지의 키
                    history_messages_key="history",  # 기록 메시지의 키
                )
            )

            # 아래부터는 질문과 응답하는 코드입니다.
            response = chain_with_memory.invoke(
                # 사용자 질문을 입력
                {"question": user_input},
                # 설정 정보로 세션 ID "abc123"을 전달합니다. 같은 세션ID를 가진 대화들은 같은 히스토리를 가집니다.
                config={"configurable": {"session_id": "abc1234"}},
            )
            # msg = response.content
            # parser를 거쳤기 때문에 컨텐츠를 따로 추출할 필요가 없음.

            # st.write(msg)
            st.session_state["messages"].append(
                ChatMessage(role="assistant", content=response)
            )
            # 세션 스테이트에 방금 대화를 추가합니다. 그래서 누적된 데이터를 입력하거나 참조해서 다음대화를 이어갈 수 있습니다.
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        st.error("파일을 업로드 해주세요.")
