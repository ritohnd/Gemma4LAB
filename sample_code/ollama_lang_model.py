from __future__ import annotations

import argparse
import asyncio
import time
from functools import lru_cache
from typing import Literal

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # pip install langchain
from langchain_ollama import ChatOllama  # pip install langchain-ollama
from pydantic import model_validator  # pip install pydantic
from pydantic_settings import (  # pip install pydantic-settings
    BaseSettings,
    SettingsConfigDict,
)

from llm.lang_model import LangModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    OLLAMA_BASE_URL: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_OLLAMA_BASE_URL() -> str:
    if "\\x3a" in get_settings().OLLAMA_BASE_URL:
        return get_settings().OLLAMA_BASE_URL.replace("\\x3a", ":")
    else:
        return get_settings().OLLAMA_BASE_URL


def get_ollama_model_list(ollama_base_url: str) -> list[str]:
    r = requests.get(f"{ollama_base_url}/api/tags")
    model_list = [m["name"] for m in r.json()["models"]]
    return model_list


class OllamaLangModel(LangModel):
    ollama_base_url: str | None = None
    ollama_model: str

    # @override
    @staticmethod
    def get_model_list(ollama_base_url: str | None = None) -> list[str]:
        return get_ollama_model_list(
            ollama_base_url=get_OLLAMA_BASE_URL() if ollama_base_url is None else ollama_base_url
        )

    @model_validator(mode="after")
    def set_ollama_base_url(self) -> OllamaLangModel:
        if self.ollama_base_url is None:
            self.ollama_base_url = get_OLLAMA_BASE_URL()
        print(f"OllamaLangModel: {self.ollama_model}({self.ollama_base_url}) will be used")
        return self

    # @override
    def get_local_model_list(self) -> list[str]:
        return get_ollama_model_list(ollama_base_url=self.ollama_base_url)

    # @override
    def _create_model(self):
        if self.reasoning_effort in ["none", "minimal"]:
            reasoning = False
        else:
            reasoning = True
        self._model = ChatOllama(
            model=self.ollama_model,
            temperature=self.temperature,
            base_url=self.ollama_base_url,
            num_predict=self.max_tokens,
            # reasoning=reasoning,
            # timeout=self.timeout,
            # max_retries=self.max_retries,
            reasoning=reasoning,
            # thinking_budget=self.thinking_budget,
            verbose=self.verbose,
            # n=self.candidate_count,
            model_kwargs={
                "keep_alive": -1  # 文字列の "-1" や "10m" も指定可能
            },
            # other params...
        )

    # @override
    def has_thoughts(self, ai_msg: AIMessage) -> bool:
        return "reasoning_content" in ai_msg.additional_kwargs

    # @override
    def get_thoughts(self, ai_msg: AIMessage) -> str:
        if self.has_thoughts(ai_msg=ai_msg):
            return ai_msg.additional_kwargs["reasoning_content"]
        return ""

    # @override
    def get_main_message(self, ai_msg: AIMessage) -> str | list:
        return ai_msg.content


def chat_test(
    model_name: str,
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal",
    include_thoughts: bool | None = None,
):
    # print(LangOllamaModel.get_model_list())
    model = OllamaLangModel(
        ollama_model=model_name,
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    start_time = time.perf_counter()
    ai_msg, response_metadata, usage_metadata = model.chat(
        system_msg="あなたは親切なアシスタントです。",
        human_msg="こんにちは",
    )
    response_time = time.perf_counter() - start_time
    print(f"response finish in {response_time:.3f}sec")
    print(f"{ai_msg=}")
    print(f"{response_metadata=}")
    print(f"{usage_metadata=}")


async def chat_astream_test(
    model_name: str,
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal",
    include_thoughts: bool | None = None,
):
    model = OllamaLangModel(
        ollama_model=model_name,
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    ai_msg = ""
    async for content, _, _ in model.chat_from_messages_astream(
        messages=[
            SystemMessage(content="あなたは親切なアシスタントです。"),
            HumanMessage(content="こんにちは"),
        ],
        print_processing_time=True,
    ):
        ai_msg += content
        print(content, end="", flush=True)
    print(f"{ai_msg=}")


async def chat_astream_sentences_test(
    model_name: str,
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal",
    include_thoughts: bool | None = None,
):
    model = OllamaLangModel(
        ollama_model=model_name,
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    ai_msg = ""
    async for content, _, _ in model.chat_from_messages_astream_sentences(
        messages=[
            SystemMessage(content="あなたは親切なアシスタントです。"),
            HumanMessage(content="こんにちは"),
        ],
        max_chars=30,
        print_processing_time=False,
    ):
        ai_msg += f"{content}"
        print(content, end="\n", flush=True)
    print(f"{ai_msg=}")


def image_recog_test(
    model_name: str,
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal",
    include_thoughts: bool | None = None,
):
    model = OllamaLangModel(
        ollama_model=model_name,
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    image_files = ["./data/image.jpg"]
    # image_files = ["./data/frame_000149.jpg"]
    start_time = time.perf_counter()
    ai_msg, response_metadata, usage_metadata = model.image_file_recognition(
        system_msg="あなたは親切なアシスタントです。",
        human_msg="この画像は何ですか？",
        image_files=image_files,
        max_length=1024,
    )
    response_time = time.perf_counter() - start_time
    print(f"response finish in {response_time:.3f}sec")
    print(f"{ai_msg=}")
    print(f"{response_metadata=}")
    print(f"{usage_metadata=}")


async def image_recog_astream_test(
    model_name: str,
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal",
    include_thoughts: bool | None = None,
):
    model = OllamaLangModel(
        ollama_model=model_name,
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    image_files = ["./data/image.jpg"]
    # image_files = ["./data/frame_000149.jpg"]
    ai_msg = ""
    async for content, _, _ in model.image_file_recognition_astream(
        system_msg="あなたは親切なアシスタントです。",
        human_msg="この画像は何ですか？",
        image_files=image_files,
        max_length=1024,
        print_processing_time=True,
    ):
        ai_msg += content
        print(content, end="", flush=True)
    print(f"{ai_msg=}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs="?", default="qwen3.5:4b")
    parser.add_argument("--image_recog", action="store_true")
    parser.add_argument("--astream", action="store_true")
    parser.add_argument("--astream-sentences", action="store_true")
    parser.add_argument("--list_models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        model_list = OllamaLangModel.get_model_list()
        print(f"Available models: {model_list}")
        return
    if args.image_recog:
        if args.astream:
            asyncio.run(
                image_recog_astream_test(
                    args.model_name,
                    temperature=0,
                    candidate_count=1,
                    reasoning_effort="minimal",
                    include_thoughts=False,
                )
            )
        else:
            image_recog_test(
                args.model_name, temperature=0, candidate_count=1, reasoning_effort="minimal", include_thoughts=False
            )
    else:
        if args.astream:
            asyncio.run(
                chat_astream_test(
                    args.model_name,
                    temperature=0,
                    candidate_count=1,
                    reasoning_effort="minimal",
                    include_thoughts=False,
                )
            )
        elif args.astream_sentences:
            asyncio.run(
                chat_astream_sentences_test(
                    args.model_name,
                    temperature=0,
                    candidate_count=1,
                    reasoning_effort="minimal",
                    include_thoughts=False,
                )
            )
        else:
            chat_test(
                args.model_name, temperature=0, candidate_count=1, reasoning_effort="minimal", include_thoughts=False
            )


if __name__ == "__main__":
    main()
