#Streamlit을 이용한 개인 문서 기반 QA 챗봇 앱
#주요 라이브러리 import 및 기능 초기화

import streamlit as st  # Streamlit 라이브러리: 웹 기반 인터페이스 구성
import tiktoken  # 텍스트를 토큰 단위로 나눌 때 사용 (ex. 길이 측정)
from loguru import logger  # 로그 출력을 위한 라이브러리

#LangChain 관련 모듈

from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인 구성
from langchain.chat_models import ChatOpenAI  # OpenAI 기반 챗 모델 사용
from langchain.document_loaders import PyPDFLoader  # PDF 문서 로더
from langchain.document_loaders import Docx2txtLoader  # DOCX 문서 로더
from langchain.document_loaders import UnstructuredPowerPointLoader  # PPTX 문서 로더
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 chunk로 분할
from langchain.embeddings import HuggingFaceEmbeddings  # 문서 임베딩 생성 도구
from langchain.memory import ConversationBufferMemory  # 대화 히스토리를 기억하는 메모리 모듈
from langchain.vectorstores import FAISS  # 유사도 검색용 벡터 저장소 (빠른 검색 가능)
from langchain.callbacks import get_openai_callback  # OpenAI 호출 비용 로깅 콜백ㅌㄹ
from langchain.memory import StreamlitChatMessageHistory  # Streamlit 전용 대화 기록 객체

#메인 함수 정의

def main():
# 페이지 기본 정보 설정
st.set_page_config(
page_title="DirChat",  # 웹 브라우저 상 탭 제목
page_icon=":books:"     # 탭 아이콘 (이모지 사용)
)

# 페이지 제목 출력 (Markdown 문법 적용됨)
st.title("_Private Data :red[QA Chat]_ :books:")

# Streamlit 세션 상태 변수 초기화 (처음 접속 시)
if "conversation" not in st.session_state:
    st.session_state.conversation = None  # 대화 체인
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None  # 대화 기록
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None  # 문서 처리 여부

# 사이드바 구성
with st.sidebar:
    # 사용자가 파일 업로드할 수 있는 위젯 (PDF, DOCX 지원)
    uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx'], accept_multiple_files=True)

    # OpenAI API 키 입력창
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

    # 문서 처리 버튼
    process = st.button("Process")

    # 처리 버튼 클릭 시 동작
    if process:
        # API 키가 입력되지 않았으면 경고 표시 후 중단
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 문서 처리: 텍스트 추출 → 청크 분할 → 벡터화
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        # 대화 체인 구성
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

# 초기 인삿말 (assistant 역할로 기본 메시지 제공)
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

# 이전 대화 메시지 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 대화 기록 객체 초기화
history = StreamlitChatMessageHistory(key="chat_messages")

# 채팅 입력 처리
if query := st.chat_input("질문을 입력해주세요."):
    # 유저 메시지 추가
    st.session_state.messages.append({"role": "user", "content": query})

    # 유저 메시지 렌더링
    with st.chat_message("user"):
        st.markdown(query)

    # 어시스턴트 응답 처리
    with st.chat_message("assistant"):
        chain = st.session_state.conversation

        with st.spinner("Thinking..."):  # 로딩 스피너 표시
            result = chain({"question": query})  # 질문 전달

            with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']  # 전체 대화 저장
                response = result['answer']  # 모델 응답 추출
                source_documents = result['source_documents']  # 참조 문서 리스트

        # 응답 출력
        st.markdown(response)

        # 참조 문서 3개까지 표시 (Expander 사용)
        with st.expander("참고 문서 확인"):
            st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
            st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
            st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

    # 어시스턴트 메시지를 대화 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

#텍스트를 토큰 개수로 측정하는 함수 (길이 계산에 사용)

def tiktoken_len(text):
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(text)
return len(tokens)

#업로드한 문서에서 텍스트 추출 함수

def get_text(docs):
doc_list = []
for doc in docs:
file_name = doc.name  # 파일 이름 추출
with open(file_name, "wb") as file:
file.write(doc.getvalue())  # 바이너리로 저장
logger.info(f"Uploaded {file_name}")

    # 파일 형식별 로더 선택
    if '.pdf' in doc.name:
        loader = PyPDFLoader(file_name)
        documents = loader.load_and_split()
    elif '.docx' in doc.name:
        loader = Docx2txtLoader(file_name)
        documents = loader.load_and_split()
    elif '.pptx' in doc.name:
        loader = UnstructuredPowerPointLoader(file_name)
        documents = loader.load_and_split()

    doc_list.extend(documents)
return doc_list

#문서를 chunk 단위로 나누는 함수

def get_text_chunks(text):
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=900,  # chunk 크기 (900자)
chunk_overlap=100,  # chunk 간 겹치는 부분
length_function=tiktoken_len  # 길이 측정 함수 지정
)
chunks = text_splitter.split_documents(text)
return chunks

#벡터 저장소 생성 함수 (임베딩 기반 유사 문서 검색용)

def get_vectorstore(text_chunks):
embeddings = HuggingFaceEmbeddings(
model_name="jhgan/ko-sroberta-multitask",  # 한국어 지원 임베딩 모델
model_kwargs={'device': 'cpu'},
encode_kwargs={'normalize_embeddings': True}
)
vectordb = FAISS.from_documents(text_chunks, embeddings)
return vectordb

#ConversationalRetrievalChain을 활용한 대화 체인 구성 함수

def get_conversation_chain(vetorestore, openai_api_key):
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
conversation_chain = ConversationalRetrievalChain.from_llm(
llm=llm,
chain_type="stuff",
retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
get_chat_history=lambda h: h,
return_source_documents=True,
verbose=True
)
return conversation_chain

#이 파일이 실행될 때 main() 함수 호출

if name == 'main':
main()

