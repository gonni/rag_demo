from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from typing import Optional, AsyncIterator
from .base import BaseChain


class TopicChain(BaseChain):
    """
    주어진 주제에 대해 설명하는 체인 클래스입니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

    def __init__(
        self,
        model: str = "exaone3.5:latest",
        temperature: float = 0,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Your mission is to explain given topic in a concise manner. Answer in Korean."
        )

    def setup(self):
        """TopicChain을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Here is the topic: {topic}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class ChatChain(BaseChain):
    """
    대화형 체인 클래스입니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

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

    def setup(self):
        """ChatChain을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class LLM(BaseChain):
    """
    기본 LLM 체인 클래스입니다.
    다른 체인들과 달리 프롬프트 없이 직접 LLM을 반환합니다.
    """

    def setup(self):
        """LLM 인스턴스를 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)
        return llm


class Translator(BaseChain):
    """
    번역 체인 클래스입니다.
    주어진 문장을 한국어로 번역합니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

    def __init__(
        self,
        model: str = "exaone3.5:latest",
        temperature: float = 0,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Your mission is to translate given sentences into Korean."
        )

    def setup(self):
        """Translator 체인을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Here is the sentence: {input}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class StreamingChatChain(BaseChain):
    """
    스트리밍 RAG 체인 클래스입니다.
    원스토어 인앱결제 관련 질문에 대해 실시간으로 응답을 스트리밍합니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
        vector_db_path (str): 벡터 데이터베이스 경로
    """

    def __init__(
        self,
        model: str = "exaone3.5:latest",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        vector_db_path: str = "models/faiss_vs_rag_iap_v7",
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or """당신은 원스토어 인앱결제 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 답변해주세요.

답변 시 다음 사항을 고려해주세요:
1. 한국어로 명확하고 이해하기 쉽게 답변하세요
2. 코드 예시가 있다면 포함해주세요
3. 단계별로 설명해주세요
4. 개발자 관점에서 실용적인 정보를 제공해주세요
5. 컨텍스트에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답변하세요"""
        )
        self.vector_db_path = vector_db_path
        self._retriever = None
        self._rag_chain = None

    def _setup_retriever(self):
        """검색기를 설정합니다."""
        if self._retriever is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                from langchain_community.vectorstores import FAISS
                
                embedding_model = OllamaEmbeddings(model="exaone3.5:latest")
                loaded_db = FAISS.load_local(
                    folder_path=self.vector_db_path,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True,
                )
                
                self._retriever = loaded_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6}
                )
            except Exception as e:
                print(f"벡터 데이터베이스 로드 실패: {e}")
                self._retriever = None

    def _setup_rag_chain(self):
        """RAG 체인을 설정합니다."""
        if self._rag_chain is None:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough
            
            # 검색기 설정
            self._setup_retriever()
            
            if self._retriever is None:
                # 검색기가 없으면 기본 채팅 체인 사용
                llm = ChatOllama(model=self.model, temperature=self.temperature)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    ("user", "{question}")
                ])
                self._rag_chain = prompt | llm | StrOutputParser()
            else:
                # RAG 체인 구성
                llm = ChatOllama(model=self.model, temperature=self.temperature)
                
                prompt_template = f"""
{self.system_prompt}

컨텍스트:
{{context}}

질문: {{question}}

답변:"""
                
                prompt = PromptTemplate.from_template(prompt_template)
                
                self._rag_chain = (
                    {"context": self._retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

    def setup(self):
        """StreamingChatChain을 설정하고 반환합니다."""
        self._setup_rag_chain()
        return self._rag_chain

    async def astream(self, messages: list) -> AsyncIterator[str]:
        """
        메시지를 받아서 스트리밍 응답을 생성합니다.
        
        Args:
            messages: 대화 메시지 리스트
            
        Yields:
            str: 스트리밍 응답 청크
        """
        self._setup_rag_chain()
        
        if self._rag_chain is None:
            yield "벡터 데이터베이스를 로드할 수 없습니다."
            return
        
        # 마지막 사용자 메시지 추출
        last_user_message = None
        for message in reversed(messages):
            if hasattr(message, 'content') and hasattr(message, '__class__'):
                if 'HumanMessage' in str(message.__class__):
                    last_user_message = message.content
                    break
        
        if not last_user_message:
            yield "사용자 메시지를 찾을 수 없습니다."
            return
        
        try:
            async for chunk in self._rag_chain.astream(last_user_message):
                if chunk:
                    yield chunk
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}"

    def invoke(self, question: str) -> str:
        """
        질문에 대한 답변을 생성합니다.
        
        Args:
            question: 질문 문자열
            
        Returns:
            str: 답변
        """
        self._setup_rag_chain()
        
        if self._rag_chain is None:
            return "벡터 데이터베이스를 로드할 수 없습니다."
        
        try:
            # 항상 dict 형태로 전달
            return self._rag_chain.invoke({"question": question})
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
