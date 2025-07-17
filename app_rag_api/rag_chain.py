from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate

from .base import BaseChain


# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content><page>{doc.metadata['page']}</page><source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )


class RagChain(BaseChain):

    def __init__(
        self,
        model: str = "exaone3.5:latest",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful AI Assistant. Your name is '고니'. You must answer in Korean."
        )
        if "file_path" in kwargs:
            self.file_path = kwargs["file_path"]

    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")

        # # Splitter 설정
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # # 문서 로드
        # loader = PDFPlumberLoader(self.file_path)
        # docs = loader.load_and_split(text_splitter=text_splitter)

        # 캐싱을 지원하는 임베딩 설정
        # EMBEDDING_MODEL = "bge-m3"
        embedding_model = OllamaEmbeddings(model="exaone3.5:latest")
        # embedding_model = OllamaEmbeddings(model="eeve-kor-10.8-KM:latest")
        # 저장된 데이터를 로드
        loaded_db = FAISS.load_local(
            folder_path="models/faiss_vs_rag_iap_touched_v6",
            # index_name="index",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

        # 문서 검색기 설정
        retriever = loaded_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 25, "lambda_mult": 0.6}
        )

        # 프롬프트 로드
        # prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")
        prompt = PromptTemplate.from_template(
            """
        당신은 한국어로 된 질문에 답변하는 AI입니다.
        주어진 질문에 대해 관련된 문서에서 정보를 찾아 답변을 작성하세요.
        답변을 생성할 수 없을 경우 '해당 질의에 대한 답변을 찾을 수 없습니다.'라고 응답하세요.
        해당 질문은 개발자들이 하는 질문으로 최대한 개발자 친화적인 답변을 생성해주세요.
        가능한한 코드 예시를 포함하여 답변을 작성해주세요.

        #Question: 
        {question} 
        #Context: 
        {context} 

        #Answer:"""
        )
        
        # Ollama 모델 지정
        llm = ChatOllama(
            model="exaone3.5:latest",
            temperature=0,
        )

        # 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
