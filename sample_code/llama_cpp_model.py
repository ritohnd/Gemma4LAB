from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from functools import lru_cache
from typing import Literal

import httpx
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import (
    ChatOpenAI,
)
from langchain_openai.chat_models.base import _convert_message_to_dict
from pydantic import PrivateAttr, model_validator
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
    LLAMA_CPP_BASE_URL: str = ""
    LLAMA_CPP_MODEL_NAME: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_LLAMA_CPP_BASE_URL() -> str:
    return get_settings().LLAMA_CPP_BASE_URL


def get_LLAMA_CPP_MODEL_NAME() -> str:
    return get_settings().LLAMA_CPP_MODEL_NAME


GEMMA4_STR_DELIM = '<|"|>'
GEMMA4_TOOL_CALL_START = "<|tool_call>call:"
GEMMA4_TOOL_CALL_END = "<tool_call|>"
GEMMA4_THOUGHT_START = "<|channel>thought\n"
# ツール利用無しを強制する際に生成を禁止するトークン
GEMMA4_FORBIDDEN_TOOL_TOKENS = ["<|tool_call>", "<|tool_response>"]
_GEMMA4_BARE_KEY_PATTERN = re.compile(r"([{,]\s*)([A-Za-z_][\w\-.]*)\s*:")


def parse_gemma4_tool_args(args_text: str) -> dict:
    """Gemma4形式の引数 (例: '{sender:<|"|>中山<|"|>,limit:3}') をdictに変換する"""
    parts = args_text.split(GEMMA4_STR_DELIM)
    if len(parts) % 2 == 0:
        raise ValueError(f"unbalanced string delimiter: {args_text!r}")
    json_text = ""
    for i, part in enumerate(parts):
        if i % 2 == 1:
            json_text += json.dumps(part, ensure_ascii=False)
        else:
            json_text += _GEMMA4_BARE_KEY_PATTERN.sub(r'\1"\2":', part)
    args = json.loads(json_text)
    if not isinstance(args, dict):
        raise ValueError(f"tool args is not an object: {args_text!r}")
    return args


def format_gemma4_tool_response(tool_name: str, content: str) -> str:
    """Gemma4チャットテンプレートの文字列ツール結果の描画 (format_tool_response_block) と同じ形式"""
    return f"<|tool_response>response:{tool_name}{{value:{GEMMA4_STR_DELIM}{content}{GEMMA4_STR_DELIM}}}<tool_response|>"


def _gbnf_literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


_GEMMA4_GBNF_STR = _gbnf_literal(GEMMA4_STR_DELIM)
_GEMMA4_ARGS_GBNF_BASE_RULES = f"""
str ::= {_GEMMA4_GBNF_STR} [^<]* {_GEMMA4_GBNF_STR}
int ::= "-"? [0-9]+
num ::= "-"? [0-9]+ ("." [0-9]+)?
bool ::= "true" | "false"
null ::= "null"
key ::= [A-Za-z_] [A-Za-z0-9_]*
value ::= str | num | bool | null | "[" (value ("," value)*)? "]" | "{{" (key ":" value ("," key ":" value)*)? "}}"
"""


def _json_schema_to_gemma4_gbnf(schema: dict) -> str:
    if "enum" in schema:
        alternatives = []
        for v in schema["enum"]:
            if isinstance(v, str):
                alternatives.append(f"{_GEMMA4_GBNF_STR} {_gbnf_literal(v)} {_GEMMA4_GBNF_STR}")
            elif v is None:
                alternatives.append("null")
            elif isinstance(v, bool):
                alternatives.append(_gbnf_literal("true" if v else "false"))
            else:
                alternatives.append(_gbnf_literal(str(v)))
        return "(" + " | ".join(alternatives) + ")"
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return "(" + " | ".join(_json_schema_to_gemma4_gbnf(s) for s in schema[key]) + ")"
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "(" + " | ".join(_json_schema_to_gemma4_gbnf({**schema, "type": t}) for t in schema_type) + ")"
    if schema_type == "array":
        item = _json_schema_to_gemma4_gbnf(schema.get("items", {}))
        return f'("[" ({item} ("," {item})*)? "]")'
    return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "null": "null"}.get(
        schema_type, "value"
    )


def build_gemma4_tool_args_grammar(parameters: dict) -> str:
    """ツール引数のJSON Schemaから、'<|tool_call>call:name{' に続く引数部分を制約するGBNFを生成する

    キー名をスキーマのものに限定するため、ツール宣言が無くても引数名を誤らない
    """
    properties = parameters.get("properties", {})
    required = [k for k in parameters.get("required", []) if k in properties]
    optional = [k for k in properties if k not in required]

    def pair(key: str) -> str:
        return f'{_gbnf_literal(key)} ":" {_json_schema_to_gemma4_gbnf(properties[key])}'

    rules = []
    required_seq = ' "," '.join(pair(k) for k in required)
    if optional:
        rules.append("opt ::= " + " | ".join(pair(k) for k in optional))
        if required:
            rules.insert(0, f'root ::= {required_seq} ("," opt)* "}}"')
        else:
            rules.insert(0, 'root ::= (opt ("," opt)*)? "}"')
    elif required:
        rules.insert(0, f'root ::= {required_seq} "}}"')
    else:
        rules.insert(0, 'root ::= "}"')
    return "\n".join(rules) + _GEMMA4_ARGS_GBNF_BASE_RULES


class LlamaCppModel(LangModel):
    llama_cpp_base_url: str | None = None
    llama_cpp_model_name: str | None = None
    # True: Gemma4のツール呼び出しをprefillで強制する (Falseならtool_choiceを使う従来方式)
    use_tool_prefill: bool = True
    tool_args_max_tokens: int = 256
    # True: 選択ツールの宣言をシステムプロンプトに含める
    # (ツールごとにプロンプト先頭が変わりKVキャッシュが効かなくなるため、既定はFalse)
    declare_selected_tool: bool = False
    # True: 引数有りツールの場合のみ選択ツールの宣言を含める (引数の意味を説明文から読み取らせるため)
    # 引数無しツールはLLMに呼び出しを生成させないので宣言不要。declare_selected_tool=Trueなら常に含める
    declare_selected_tool_with_args: bool = False
    # True: 引数生成をツールのスキーマから生成したGBNFで制約する
    constrain_tool_args: bool = True
    _forbidden_tool_logit_bias: dict[str, float] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_base_url_and_model_name(self) -> LlamaCppModel:
        if self.llama_cpp_base_url is None:
            self.llama_cpp_base_url = get_LLAMA_CPP_BASE_URL()
        if self.llama_cpp_model_name is None:
            self.llama_cpp_model_name = get_LLAMA_CPP_MODEL_NAME()
        return self

    # @override
    @staticmethod
    def get_model_list() -> list[str]:
        return [get_LLAMA_CPP_MODEL_NAME()]

    # @override
    def _create_model(self):
        if self.reasoning_effort == "none":
            self._model = ChatOpenAI(
                model=self.llama_cpp_model_name,
                api_key="local",
                base_url=f"{self.llama_cpp_base_url}/v1",
                temperature=self.temperature,
                timeout=self.timeout,
                max_completion_tokens=self.max_tokens,
                max_retries=self.max_retries,
                reasoning_effort=self.reasoning_effort,
                verbose=self.verbose,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                streaming=True,
                stream_usage=True,
            )
        else:
            self._model = ChatOpenAI(
                model=self.llama_cpp_model_name,
                api_key="local",
                base_url=f"{self.llama_cpp_base_url}/v1",
                temperature=self.temperature,
                timeout=self.timeout,
                max_completion_tokens=self.max_tokens,
                max_retries=self.max_retries,
                reasoning_effort=self.reasoning_effort,
                verbose=self.verbose,
                streaming=True,
                stream_usage=True,
            )
        # 推論時に毎回取得しないよう、モデル生成時に禁止トークンIDを一度だけ取得しておく
        self._forbidden_tool_logit_bias = self._fetch_forbidden_tool_logit_bias()

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
        # print(f"{ai_msg=}")
        return ai_msg.content

    def _enable_thinking(self) -> bool:
        return self.reasoning_effort != "none"

    def _apply_template(self, messages: list[BaseMessage], tools: list[dict]) -> str:
        body = {
            "messages": [_convert_message_to_dict(m) for m in messages],
            "tools": tools,
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking()},
        }
        res = httpx.post(f"{self.llama_cpp_base_url}/apply-template", json=body, timeout=self.timeout)
        res.raise_for_status()
        return res.json()["prompt"]

    def _raw_completion(
        self, prompt: str, max_tokens: int | None, stop: list[str] | None = None, grammar: str | None = None
    ) -> dict:
        body = {
            "prompt": prompt,
            "temperature": self.temperature,
            "n_predict": max_tokens if max_tokens is not None else -1,
            "cache_prompt": True,
        }
        if stop:
            body["stop"] = stop
        if grammar:
            body["grammar"] = grammar
        res = httpx.post(f"{self.llama_cpp_base_url}/completion", json=body, timeout=self.timeout)
        res.raise_for_status()
        return res.json()

    def _fetch_forbidden_tool_logit_bias(self) -> dict[str, float] | None:
        logit_bias: dict[str, float] = {}
        try:
            for token in GEMMA4_FORBIDDEN_TOOL_TOKENS:
                res = httpx.post(
                    f"{self.llama_cpp_base_url}/tokenize",
                    json={"content": token, "parse_special": True},
                    timeout=self.timeout,
                )
                res.raise_for_status()
                token_ids = res.json()["tokens"]
                # 単一の特殊トークンの場合のみ禁止する (通常テキストのトークンを巻き込まないため)
                if len(token_ids) == 1:
                    logit_bias[str(token_ids[0])] = -100
                else:
                    print(f"{token!r} is not a single special token, skip forbidding")
        except httpx.HTTPError as e:
            # サーバー未起動などで取得できなかった場合は、初回推論時に再取得する
            print(f"failed to fetch forbidden tool token ids: {e}")
            return None
        return logit_bias

    def _get_forbidden_tool_logit_bias(self) -> dict[str, float]:
        if self._forbidden_tool_logit_bias is None:
            self._forbidden_tool_logit_bias = self._fetch_forbidden_tool_logit_bias()
        return self._forbidden_tool_logit_bias or {}

    # @override
    def chat_from_messages_without_tool(
        self, messages: list[BaseMessage]
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        """ツール呼び出しトークン(<|tool_call>等)の生成をlogit_biasで禁止して応答を生成する"""
        if self.verbose:
            start_time = time.perf_counter()
        model_without_tool = self._model.bind(logit_bias=self._get_forbidden_tool_logit_bias())
        prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
        ai_msg = model_without_tool.invoke(prompt)
        if self.verbose:
            response_time = time.perf_counter() - start_time
            print(f"response finish in {response_time:.3f}sec")
        return self.parse_ai_msg(ai_msg=ai_msg)

    # @override
    def chat_from_messages_with_tool_choice(
        self, messages: list[BaseMessage], tool: BaseTool, update_messages: bool = True
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        """Gemma4前提: ツール呼び出しをprefillで強制してから最終回答を生成する

        - 引数無しツール: LLMを呼ばずにツールを実行し、ツール呼び出し+実行結果までをprefillして続きを生成
        - 引数有りツール: '<|tool_call>call:{tool名}{' までをprefillして引数のみを生成させ、
          ツール実行後に同様に続きを生成

        テンプレート適用は1回のみで、ツール呼び出し・実行結果はGemma4形式の文字列として連結する
        """
        if not self.use_tool_prefill:
            return super().chat_from_messages_with_tool_choice(
                messages=messages, tool=tool, update_messages=update_messages
            )
        if self.verbose:
            start_time = time.perf_counter()
        openai_tool = convert_to_openai_tool(tool)
        input_tokens = 0
        output_tokens = 0

        declare_tool = self.declare_selected_tool or (self.declare_selected_tool_with_args and bool(tool.args))
        prompt = self._apply_template(messages=messages, tools=[openai_tool] if declare_tool else [])
        prompt += f"{GEMMA4_TOOL_CALL_START}{tool.name}{{"
        if tool.args:
            grammar = (
                build_gemma4_tool_args_grammar(openai_tool["function"].get("parameters", {}))
                if self.constrain_tool_args
                else None
            )
            res = self._raw_completion(
                prompt=prompt, max_tokens=self.tool_args_max_tokens, stop=[GEMMA4_TOOL_CALL_END], grammar=grammar
            )
            input_tokens += res.get("tokens_evaluated", 0)
            output_tokens += res.get("tokens_predicted", 0)
            args_body = res["content"]
            try:
                tool_args = parse_gemma4_tool_args("{" + args_body.strip())
            except ValueError as e:
                print(f"failed to parse tool args ({e}), fallback to chat without tool")
                return self.chat_from_messages(messages=messages)
        else:
            args_body = "}"
            tool_args = {}

        tool_call = {"name": tool.name, "args": tool_args, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "tool_call"}
        if self.verbose:
            print(f"tool_call: {tool_call['name']}, args: {tool_call['args']}")
        tool_result = str(tool.invoke(tool_args))
        if update_messages:
            messages.append(AIMessage(content="", tool_calls=[tool_call]))
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"], name=tool.name))

        # テンプレート同様<tool_response|>の直後(=モデルターンの途中)から続きを生成させる
        # 生成済みの引数テキストをそのまま使うので、引数生成時のKVキャッシュがそのまま再利用される
        final_prompt = (
            prompt + args_body + GEMMA4_TOOL_CALL_END + format_gemma4_tool_response(tool.name, tool_result)
        )
        if self._enable_thinking():
            final_prompt += GEMMA4_THOUGHT_START
        res = self._raw_completion(prompt=final_prompt, max_tokens=self.max_tokens)
        input_tokens += res.get("tokens_evaluated", 0)
        output_tokens += res.get("tokens_predicted", 0)
        if self.verbose:
            response_time = time.perf_counter() - start_time
            print(f"response finish in {response_time:.3f}sec")

        response_metadata = {
            "model_name": self.llama_cpp_model_name,
            "stop_type": res.get("stop_type"),
            "tool_call": tool_call,
        }
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return (res["content"].strip(), response_metadata, usage_metadata)


def chat_test(
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none",
    include_thoughts: bool | None = None,
):
    model = LlamaCppModel(
        llama_cpp_model_name="gemma-4-E4B-it-qat",
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    start_time = time.perf_counter()
    ai_msg, response_metadata, usage_metadata = model.chat(
        system_msg="",
        human_msg="こんにちは！自己紹介してください。",
    )
    response_time = time.perf_counter() - start_time
    print(f"response finish in {response_time:.3f}sec")
    print(f"{ai_msg=}")
    print(f"{response_metadata=}")
    print(f"{usage_metadata=}")


async def chat_astream_test(
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none",
    include_thoughts: bool | None = None,
):
    model = LlamaCppModel(
        llama_cpp_model_name="gemma-4-E4B-it-qat",
        temperature=temperature,
        candidate_count=candidate_count,
        reasoning_effort=reasoning_effort,
        include_thoughts=include_thoughts,
    )
    model.create_model()
    ai_msg = ""
    async for content, _, _ in model.chat_astream(
        system_msg="",
        human_msg="こんにちは！自己紹介してください。",
        # system_msg="あなたは親切なアシスタントです。",
        # human_msg="こんにちは",
        print_processing_time=True,
    ):
        ai_msg += content
        print(content, end="", flush=True)
    print(f"{ai_msg=}")


def image_recog_test(
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none",
    include_thoughts: bool | None = None,
):
    model = LlamaCppModel(
        llama_cpp_model_name="gemma-4-E4B-it-qat",
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
    print(f"{ai_msg=}")
    print(f"response finish in {response_time:.3f}sec")
    print(f"{response_metadata=}")
    print(f"{usage_metadata=}")


async def image_recog_astream_test(
    temperature: float = 0,
    candidate_count: int = 1,
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none",
    include_thoughts: bool | None = None,
):
    model = LlamaCppModel(
        llama_cpp_model_name="gemma-4-E4B-it-qat",
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
    parser.add_argument("model_name", nargs="?", default="gemma-4-E4B-it-qat")
    parser.add_argument("--image_recog", action="store_true")
    parser.add_argument("--astream", action="store_true")
    args = parser.parse_args()
    if args.image_recog:
        if args.astream:
            asyncio.run(
                image_recog_astream_test(
                    temperature=0,
                    candidate_count=1,
                    # reasoning_effort="medium",
                    reasoning_effort="none",
                    include_thoughts=True,
                )
            )
        else:
            image_recog_test(
                temperature=0,
                candidate_count=1,
                # reasoning_effort="medium",
                reasoning_effort="none",
                include_thoughts=True,
            )
    else:
        if args.astream:
            asyncio.run(
                chat_astream_test(
                    temperature=0,
                    candidate_count=1,
                    reasoning_effort="none",
                    include_thoughts=False,
                )
            )
        else:
            chat_test(
                temperature=0,
                candidate_count=1,
                reasoning_effort="none",
                include_thoughts=False,
            )


if __name__ == "__main__":
    main()
