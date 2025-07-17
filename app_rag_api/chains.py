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
    스트리밍 대화형 체인 클래스입니다.
    실시간으로 응답을 스트리밍합니다.

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
        """StreamingChatChain을 설정하고 반환합니다."""
        llm = ChatOllama(
            model=self.model, 
            temperature=self.temperature
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain

    async def astream(self, messages: list) -> AsyncIterator[str]:
        """
        메시지를 받아서 스트리밍 응답을 생성합니다.
        
        Args:
            messages: 대화 메시지 리스트
            
        Yields:
            str: 스트리밍 응답 청크
        """
        chain = self.setup()
        
        async for chunk in chain.astream({"messages": messages}):
            if chunk:
                yield chunk
