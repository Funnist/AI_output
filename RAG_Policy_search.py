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

from operator import itemgetter


st.set_page_config(page_title="정책서 검색기", page_icon="💬")
st.title("정책서 검색기")

# 정책서 DB데이터의 위치입니다.
DB_PATH = "./vectorstores/db/"

# 데이터의 임베딩을 설정합니다. 허깅페이스는 AI계의 github라고 불리는 곳입니다. AI 관련 오픈소스는 여기에 데이터가 가장 많습니다.
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-multitask",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
    # GPU Device 설정:
    # 저는 M1 맥북을 사용하기 때문에 mps로 설정합니다. 다른 서버역할을 하는 곳의 하드웨어에 맞추어서 설정을 변경해 주셔야 합니다.
    # DBstore.py의 embedding 설정도 동일하게 맞추어 주셔야 정상적으로 동작합니다. 싱크를 꼭 맞춰주셔요.
    # - NVidia GPU: "cuda"
    # - Mac M1, M2, M3: "mps"
    # - CPU: "cpu"
)

# 로컬에 저장된 임베딩 파일을 읽어옵니다.

vectorstore = FAISS.load_local(
    DB_PATH, embeddings, allow_dangerous_deserialization=True
)

# 저장된 데이터의 검색기를 선언합니다.
retriever = vectorstore.as_retriever()

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


# 채팅 입력
if user_input := st.chat_input("무엇이 궁금하신가요?"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # 채팅 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # Stream handler를 설정하지 않으면 문장을 모두 완성한 후에 한번에 입력됩니다. 그래서 사용자가 딜레이를 더 크게 느끼게 됩니다.
        # 설정한다면, 음절단위로 출력하게끔 되어서 좀 더 일찍 답변을 받아볼 수 있게 됩니다.

        # LLM 모델설정을 설정합니다. ollama 의 로컬 LLM을 사용하기 때문에 미리 백그라운드에 설정이 필요합니다.
        # 이번에 사용한 모델은 llama3.1에 한글을 파인튜닝한 모델입니다.
        llm = ChatOllama(
            model="llama3.2_ko:latest",
            streaming=True,
            callbacks=[stream_handler],
        )

        # 아래는 프롬프트로 이것만 수정해도 LLM 답변의 질이 높아 질 수 있습니다.
        prompt = PromptTemplate.from_template(
            """너는 컨텍스트를 기반으로 해서 질문에 대한 답변을 하는 챗봇이야. 
            Use the following pieces of retrieved context to answer the question. 
            i, ii, iii 와 같은 프리픽스는 무시해도 좋아.
            답변의 길이가 길 경우에는, 가독성이 좋게 정리를 부탁해.
            If you don't know the answer, just say that you don't know. 
            Please answer in Korean.

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
