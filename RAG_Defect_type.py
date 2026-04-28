import os

# from uuid import UUID
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

st.set_page_config(page_title="결함 유형 체크리스트 검색", page_icon="💬")
st.title("결함 유형 체크리스트 검색")

DB_PATH = "./vectorstores/D_type_search_db/"
# DB_PATH = "./vectorstores/defect_type_check_list_db/"

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    # model_name="google/tapas-large-finetuned-wtq",
    # model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "mps"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local(
    DB_PATH, embeddings, allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


# 채팅 히스토리 출력
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 채팅 히스토리 기억
if "store" not in st.session_state:
    st.session_state["store"] = dict()


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
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


with st.sidebar:

    if st.button("대화기록 초기화"):
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.experimental_rerun()

# streamlit은 계속 처음부터 다시 그리는 특징이 있다. 그래서 대화상태를 저장해서 계속 뿌려주는 코드가 필요하다.
# 현재 위에 RAG값이나 밑에 채팅에서 retriever를 못찾는것도 채팅을 입력할때 처음부터 다시 시작하고 버튼을 눌렀을때의 스테이트가 사라져서 그런것으로 추측된다.
# 이래서 예제들에 버튼을 누르는 것에서 리트리버 생성, 체인, 답변생성까지 한큐로 가는것 같다. 위의 사이드바 버튼 누르는 코드를 채팅쪽으로 넣는 방법을 연구해야 되겠다.
# 다른예제를 보면 리트리버 엔진을 만들어서 세션의 일부분으로 저장하고 있는것처럼 보였다. 이것에 대한 연구도 필요하다.

# 채팅 입력
if user_input := st.chat_input("무엇이 궁금하신가요?"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # 채팅 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # 모델설정
        llm = ChatOllama(
            model="llama3.1_ko:latest", streaming=True, callbacks=[stream_handler]
        )

        prompt = PromptTemplate.from_template(
            """너는 제목과 그 하위 세부내역을 정리하는 챗봇이야.
            사용자가 입력한 질문을 바탕으로 "제목 :" 라인의 내용 중에 유사한 내용들을 검색하고 찾았다면, 하위 세부내역을 정리해서 보여줘.
            검색된 내용이 여러개라면 각각 따로 정리를 부탁해
            If you don't know the answer, just say that you don't know.
            Answer in Korean.

            "[Internal] Employee and Partner Only" 문구는 무시해도 좋아

            사용자의 질문을 "제목 :"으로 시작하는 라인의 내용과 비교해서, 유사한 "이슈"들을 검색하고 유사한 내용이 검색된다면, 이슈의 세부내역을 함께 요약해줘.

            아래 형식으로 부탁할께.

            이슈 :
            이슈의 세부내역 :

            #Previous Chat History:
            {history}

            #Question:
            {question}

            #Context:
            {context}

            #Answer:
            """
        )
        # prompt = PromptTemplate.from_template(
        #     """너는 오직 주어진 컨텍스트를 기반으로 해서 질문에 답하는 쳇봇이야.
        #     질문에 대한 답을 여러개를 찾는다면 각각 따로 요약 부탁해
        #     1) 2) 3)과 같이 여러개의 답을 출력한다면, 엔터로 구분해줘.

        #     If you don't know the answer, just say that you don't know.
        #     Answer in Korean.

        #     #Previous Chat History:
        #     {history}

        #     #Question:
        #     {question}

        #     #Context:
        #     {context}

        #     #Answer:
        #     """
        # )
        # retriever = embed_file()

        # 체인을 생성합니다.
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
        chain_with_memory = (
            RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                chain,  # 실행할 chain 객체
                get_session_history,  # 세션 기록을 가져오는 함수
                input_messages_key="question",  # 입력 메시지의 키
                history_messages_key="history",  # 기록 메시지의 키
            )
        )

        response = chain_with_memory.invoke(
            # 사용자 질문을 입력
            {"question": user_input},
            # 설정 정보로 세션 ID "abc123"을 전달합니다.
            config={"configurable": {"session_id": "abc1234"}},
        )
        # msg = response.content
        # parser를 거쳤기 때문에 컨텐츠를 따로 추출할 필요가 없음.

        # st.write(msg)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )

        # 임베딩해결,, 리트리버를 통해서 답변을 받아내는걸 구현하면 완료
        # plan B로 파인튜닝도 알아볼 필요가 있을것 같아. 허깅페이스 임베딩을 위해 인터넷 사용이 필요하다면, 로컬사용에 의미가 없어짐. 이걸 미리 db로 저장하는 형식으로 좀 해결봄
