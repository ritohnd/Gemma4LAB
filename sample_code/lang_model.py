import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import Literal

import cv2  # pip install opencv-python
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # pip install langchain
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools.base import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, PrivateAttr  # pip install pydantic


def convert_image_to_base64(image_path: str) -> bytes:
    # 拡張子を取得して小文字に変換
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        # PNG → JPEG に変換
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        success, encoded_image = cv2.imencode(".jpg", img_rgb)
        if success:
            image_bytes = encoded_image.tobytes()
    elif ext in [".jpg", ".jpeg"]:
        # JPEGはそのまま読み込む
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    else:
        raise ValueError("only .png, .jpg, .jpeg are supported")

    # Base64に変換
    return base64.b64encode(image_bytes).decode("utf-8")


def convert_matlike_to_base64(image: cv2.typing.MatLike) -> bytes:
    # 1. メモリ上でJPEG形式にエンコードする
    # cv2.imencodeの戻り値は (成功フラグ, エンコードされたバイナリ配列)
    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("JPEGへのエンコードに失敗しました。")

    # 2. バイナリデータをBase64にエンコードし、文字列(utf-8)に変換する
    base64_str = base64.b64encode(encoded_image).decode("utf-8")

    return base64_str


def convert_mp4_to_base64(video_path: str) -> bytes:
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def resize_image_file(
    src_image_path: str,
    dst_image_path: str,
    max_length: int = 1024,
    jpeg_quality: int = 80,
):
    src_image = cv2.imread(src_image_path)
    height, width = src_image.shape[:2]
    if height >= width and height > max_length:
        new_height = max_length
        new_width = int(width * (max_length / height))
    elif width > height and width > max_length:
        new_width = max_length
        new_height = int(height * (max_length / width))
    else:
        print(f"resize is not needed. {height=}, {width=}")
        cv2.imwrite(dst_image_path, src_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return
    resized_image = cv2.resize(src_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(dst_image_path, resized_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def resize_image(
    src_image: cv2.typing.MatLike,
    max_length: int = 1024,
    interpolation: int = cv2.INTER_LANCZOS4,
    verbose: bool = False,
) -> cv2.typing.MatLike:
    height, width = src_image.shape[:2]
    if height >= width and height > max_length:
        new_height = max_length
        new_width = int(width * (max_length / height))
    elif width > height and width > max_length:
        new_width = max_length
        new_height = int(height * (max_length / width))
    else:
        if verbose:
            print(f"resize is not needed. {height=}, {width=}")
        return src_image
    if verbose:
        print(f"resize_image {width}x{height} -> {new_width}x{new_height}")
    resized_image = cv2.resize(src_image, (new_width, new_height), interpolation=interpolation)
    return resized_image


def resize_image_files(
    src_image_files: list[str],
    dst_image_dir: str,
    max_length: int = 1024,
    jpeg_quality: int = 80,
) -> list[str]:
    resized_image_files = []
    for image_file in src_image_files:
        resized_image_file = os.path.join(dst_image_dir, os.path.basename(image_file))
        if os.path.splitext(resized_image_file)[1] == ".png":
            resized_image_file = os.path.splitext(resized_image_file)[0] + ".jpg"
        resize_image_file(
            src_image_path=image_file,
            dst_image_path=resized_image_file,
            max_length=max_length,
            jpeg_quality=jpeg_quality,
        )
        resized_image_files.append(resized_image_file)
    return resized_image_files


def resize_images(
    src_images: list[cv2.typing.MatLike],
    max_length: int = 1024,
    interpolation: int = cv2.INTER_LANCZOS4,
    verbose: bool = False,
) -> list[cv2.typing.MatLike]:
    resized_images = []
    for src_image in src_images:
        resized_images.append(
            resize_image(src_image=src_image, max_length=max_length, interpolation=interpolation, verbose=verbose)
        )
    return resized_images


def parse_invoke_events(events: dict, chat_state: dict, verbose: bool = False) -> tuple[str, str]:
    final_ai_msg = ""
    process_history = ""
    for i, ai_message in enumerate(events["messages"]):
        if i < len(chat_state["messages"]) - 1:
            continue
        # ユーザーの入力
        if ai_message.type == "human":
            process_history += f"- User: {ai_message.content}"
            if verbose:
                print(f"\n👤 User: {ai_message.content}")

        else:
            # AIの思考 or ツール呼び出し要求
            if ai_message.type == "ai":
                if ai_message.tool_calls:
                    for tool_call in ai_message.tool_calls:
                        process_history += f"\n\n- Agent: ツール呼び出しを決定: 「{tool_call['name']}」"
                        process_history += f"\n    引数: {tool_call['args']}"
                        if verbose:
                            print(f"\n🧠 Agent: ツール呼び出しを決定: 「{tool_call['name']}」")
                            print(f"    引数: {tool_call['args']}")
                elif ai_message.content:
                    final_ai_msg = ai_message.content
                    process_history += "\n\n- Agent: 最終回答生成"
                    process_history += f"\n   -> {ai_message.content}"
                    if verbose:
                        print("\n🧠 Agent: 最終回答生成")
                        print(f"   ➔ {ai_message.content}")
                        print("\n" + "=" * 55)
            # ツールの実行結果
            elif ai_message.type == "tool":
                try:
                    parsed = (
                        json.loads(ai_message.content) if isinstance(ai_message.content, str) else ai_message.content
                    )
                    tool_result = json.dumps(parsed, indent=4, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    tool_result = str(ai_message.content)
                tool_result_indented = "\n".join("     " + line for line in tool_result.splitlines())
                process_history += "\n\n- System: ツール実行"
                process_history += f"\n   -> 結果:\n{tool_result_indented}"
                if verbose:
                    print("\n⚙️  System: ツール実行")
                    print(f"   ➔ 結果:\n{tool_result_indented}")
            chat_state["messages"].append(ai_message)
    return (final_ai_msg, process_history)


def parse_stream_events(events: Iterator, chat_state: dict, verbose: bool = False) -> tuple[str, str]:
    final_ai_msg = ""
    process_history = ""
    if verbose:
        print(f"\n👤 User: {chat_state['messages'][-1].content}")
    for event in events:
        # event は {"ノード名": {"messages": [...]}} という辞書構造になっています
        for node_name, data in event.items():
            # 1. Gemma 4 が思考・判断している時
            if node_name == "agent":
                # 直近のAIのメッセージを取得
                ai_message = data["messages"][-1]

                if ai_message.tool_calls:
                    if ai_message.content:
                        process_history += f"   {ai_message.content}"
                        if verbose:
                            print(f"   {ai_message.content}")
                    for tool_call in ai_message.tool_calls:
                        process_history += f"\n- Agent: ツール呼び出しを決定: 「{tool_call['name']}」"
                        process_history += f"\n    引数: {tool_call['args']}"
                        if verbose:
                            print(f"\n🧠 Agent: ツール呼び出しを決定: 「{tool_call['name']}」")
                            print(f"    引数: {tool_call['args']}")
                elif ai_message.content:
                    final_ai_msg = ai_message.content
                    process_history += "\n\n- Agent: 最終回答生成"
                    process_history += f"\n   -> {ai_message.content}"
                    if verbose:
                        print("\n🧠 Agent: 最終回答生成")
                        print(f"   ➔ {ai_message.content}")
                        print("\n" + "=" * 55)

            # 2. Pythonのツール（関数）が実行されている時
            elif node_name == "tools":
                # 直近のツール実行結果メッセージを取得
                tool_message = data["messages"][-1]
                process_history += "\n\n- System: ツール実行"
                process_history += f"\n   -> 結果: {tool_message.content}"
                if verbose:
                    print("\n⚙️  System: ツール実行")
                    print(f"   ➔ 結果: {tool_message.content}")
            chat_state["messages"].extend(data["messages"])
    return (final_ai_msg, process_history)


class AIMessageBuffer(BaseModel):
    """
    逐次入力されるテキストを蓄積し、
    文末記号まで到達した文を順番に取り出す。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    delimiters: str = "。！？!?\n"
    trailing_characters: str = "😊😄😁😂😋👍✨🙏🎉🛍️♪"
    fallback_delimiters: str = "、，,"
    max_chars: int | None = None
    _buffer: str = PrivateAttr(default="")

    def append(self, chunk: str) -> None:
        """LLMから受け取ったテキスト断片を追加する。"""
        if not isinstance(chunk, str):
            raise TypeError(f"chunk must be str, got {type(chunk).__name__}")
        if chunk:
            self._buffer += chunk

    def pop(self) -> str | None:
        """
        完成した文を1つ取り出す。
        文末記号の直後に通常文字が存在する場合はその通常文字より前までを完成文として取り出す。
        バッファが文末記号や絵文字で終了している場合は続きが来るまで確定を待つ。
        max_charsに達した場合は読点などを優先して強制的に分割する。
        """
        if not self._buffer:
            return None
        split_index = self._find_sentence_end()
        if split_index is not None:
            return self._take(split_index)
        if self.max_chars is not None and len(self._buffer) >= self.max_chars:
            split_index = self._find_fallback_split_index(
                limit=self.max_chars,
            )
            return self._take(split_index)
        return None

    def pop_all(self) -> list[str]:
        """
        現時点で完成している文をすべて取り出す。
        単体テストや同期処理向け。
        ストリーム処理では通常pop()を使用する。
        """
        sentences: list[str] = []
        while (sentence := self.pop()) is not None:
            sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        """
        ストリーム終了時に残っているテキストを取り出す。
        """
        text = self._buffer.strip()
        self._buffer = ""
        return text or None

    def clear(self) -> None:
        """
        未確定テキストを破棄する。
        発話割り込みやキャンセル時に使用する。
        """
        self._buffer = ""

    def _find_sentence_end(self) -> int | None:
        """
        完成した文の終了位置を返す。
        文末記号の後ろに連続する文末記号や絵文字を吸収し、その次の通常文字が到着した時点で文末を確定する。
        バッファ末尾まで文末記号または装飾文字が続いている場合は後続文字を待つためNoneを返す。
        """
        delimiter_set = set(self.delimiters)
        trailing_set = set(self.trailing_characters)

        for index, character in enumerate(self._buffer):
            if character not in delimiter_set:
                continue
            end = index + 1
            # 文末記号の後ろに続く記号や絵文字を吸収する
            while end < len(self._buffer):
                next_character = self._buffer[end]
                if next_character in delimiter_set or next_character in trailing_set:
                    end += 1
                    continue
                # 通常文字が到着したので、
                # その直前までを完成文とする
                return end
            # バッファ末尾まで到達した場合、
            # まだ絵文字などが続く可能性があるので確定しない
            return None
        return None

    def _find_fallback_split_index(
        self,
        limit: int,
    ) -> int:
        """
        limit以内にある最後の読点または空白で分割する。
        分割候補がなければlimitで強制分割する。
        """
        search_target = self._buffer[:limit]
        delimiter_set = set(self.fallback_delimiters)
        for index in range(len(search_target) - 1, -1, -1):
            if search_target[index] in delimiter_set:
                return index + 1
        return limit

    def _take(
        self,
        split_index: int,
    ) -> str | None:
        """
        指定位置までをバッファから取り出す。
        """
        text = self._buffer[:split_index].strip()
        self._buffer = self._buffer[split_index:]
        return text or None


class LangModel(BaseModel):
    temperature: float = 0.0
    max_tokens: int | None = None
    max_retries: int = 1
    timeout: float | None = None
    verbose: bool = True
    include_thoughts: bool | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "minimal"
    candidate_count: int = 1
    input_tokens_tag: str = "input_tokens"
    output_tokens_tag: str = "output_tokens"
    do_warmup: bool = True
    _model: BaseChatModel | None = PrivateAttr(default=None)
    _agent: CompiledStateGraph | None = PrivateAttr(default=None)

    # abstract method
    @staticmethod
    def get_model_list() -> list[str]:
        return []

    @staticmethod
    def get_process_history(chat_state: dict) -> str:
        process_history = ""
        messages = chat_state.get("messages", [])
        idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].type == "human"),
            None,  # 見つからなかった場合のデフォルト値（省略すると見つからない場合に StopIteration）
        )
        if idx is not None:
            last_messages = messages[idx:]
        else:
            last_messages = messages
        for message in last_messages:
            if message.type == "human":
                process_history += f"\n\n- User: {message.content}"
            elif message.type == "ai":
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        process_history += "\n\n- Agent: ツール実行:"
                        process_history += f"\n    関数: {tool_call['name']}"
                        process_history += f"\n    引数: {tool_call['args']}"
                elif message.content:
                    process_history += "\n\n- Agent: 最終回答生成"
                    process_history += f"\n   -> {message.content}"
            elif message.type == "tool":
                try:
                    parsed = json.loads(message.content) if isinstance(message.content, str) else message.content
                    tool_result = json.dumps(parsed, indent=4, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    tool_result = str(message.content)
                tool_result_indented = "\n".join(line for line in tool_result.splitlines())
                process_history += f"\n    -> 結果: {tool_result_indented}"
        return process_history

    # abstract method
    def get_local_model_list(self) -> list[str]:
        return []

    def create_model(self):
        self._create_model()
        if self.do_warmup:
            self._warmup()

    def _warmup(self):
        if self._model is not None:
            self.chat(system_msg="あなたは親切なアシスタントです。", human_msg="こんにちは")

    # abstract method
    def _create_model(self):
        pass

    def create_agent(self, tools: list[BaseTool], prompt: SystemMessage | str):
        self._agent = create_agent(model=self._model, tools=tools, system_prompt=prompt)

    def stream_agent(
        self,
        chat_state: dict,  # {"messages": list[HumanMessage | AIMessage | ToolMessage]}
        verbose: bool = False,
    ) -> tuple[str, str]:  # (final_ai_msg, process_history)
        events = self._agent.stream(chat_state, stream_mode="updates")
        final_ai_msg, process_history = parse_stream_events(events=events, chat_state=chat_state, verbose=verbose)
        return (final_ai_msg, process_history)

    def invoke_agent(
        self,
        chat_state: dict,  # {"messages": list[HumanMessage | AIMessage | ToolMessage]}
        config: RunnableConfig | None = None,
        verbose: bool = False,
    ) -> tuple[str, str]:  # (final_ai_msg, process_history)
        events = self._agent.invoke(chat_state, config=config)
        final_ai_msg, process_history = parse_invoke_events(events=events, chat_state=chat_state, verbose=verbose)
        return (final_ai_msg, process_history)

    def get_simple_chat_chain(
        self,
        system_msg: str,
    ):
        system_message = SystemMessage(content=system_msg)
        human_message_template = HumanMessagePromptTemplate.from_template("{user_input}")
        prompt = ChatPromptTemplate.from_messages([system_message, human_message_template])
        chain = prompt | self._model
        return chain

    @staticmethod
    def get_messages(
        system_msg: str,
        human_msg: str,
        conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
    ) -> list[BaseMessage]:
        if conversation_history is not None and len(conversation_history) % 2 == 1:
            print("len(conversation) must be an even number")
            return []
        messages: list[BaseMessage] = []
        if system_msg:
            messages.append(SystemMessage(content=system_msg))
        if conversation_history is not None:
            for index, msg in enumerate(conversation_history):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(AIMessage(content=msg))
        messages.append(HumanMessage(content=human_msg))
        return messages

    @staticmethod
    def get_chat_prompt_from_messages(messages: list[BaseMessage]) -> PromptValue:
        template = ChatPromptTemplate.from_messages(messages)
        return template.invoke({})

    # abstract method
    def has_thoughts(self, ai_msg: AIMessage) -> bool:
        return False

    # abstract method
    def get_thoughts(self, ai_msg: AIMessage) -> str:
        return ""

    # abstract method
    def get_main_message(self, ai_msg: AIMessage) -> str | list:
        return ""

    def parse_ai_msg(self, ai_msg: AIMessage) -> tuple[str | list, dict, dict]:
        # print(f"{ai_msg=}")
        response_metadata = ai_msg.response_metadata
        if self.include_thoughts and self.has_thoughts(ai_msg=ai_msg):
            # thinkingをcontentからmetadataに移す
            response_metadata["thinking"] = self.get_thoughts(ai_msg=ai_msg)
        return (self.get_main_message(ai_msg=ai_msg), response_metadata, ai_msg.usage_metadata)

    def invoke_from_prompt(
        self, prompt: PromptValue, verbose: bool = False
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        if verbose:
            start_time = time.perf_counter()
        ai_msg = self._model.invoke(prompt)
        if verbose:
            response_time = time.perf_counter() - start_time
            print(f"response finish in {response_time:.3f}sec")
        return self.parse_ai_msg(ai_msg=ai_msg)

    async def astream_from_prompt(
        self,
        prompt: PromptValue,
    ) -> AsyncIterator[tuple[str | list, dict, dict]]:  # [content, response_metadata, usage_metadata]
        async for chunk in self._model.astream(prompt):
            # print(f"{chunk=}")
            content, response_metadata, usage_metadata = self.parse_ai_msg(ai_msg=chunk)
            if not content and not response_metadata and not usage_metadata:
                continue
            yield (content, response_metadata, usage_metadata)

    def chat(
        self,
        system_msg: str,
        human_msg: str,
        conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        messages = LangModel.get_messages(
            system_msg=system_msg,
            human_msg=human_msg,
            conversation_history=conversation_history,
        )
        prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
        return self.invoke_from_prompt(prompt=prompt)

    def chat_from_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
        return self.invoke_from_prompt(prompt=prompt, verbose=self.verbose)

    def chat_from_messages_without_tool(
        self, messages: list[BaseMessage]
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        # ツール呼び出しの生成を禁止したい場合用 (対応していないモデルでは通常のchatと同じ)
        return self.chat_from_messages(messages=messages)

    def chat_from_messages_with_tool_choice(
        self, messages: list[BaseMessage], tool: BaseTool, update_messages: bool = True
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        if self.verbose:
            start_time = time.perf_counter()
        model_with_tool_choice = self._model.bind_tools(
            [tool],
            tool_choice={"type": "function", "function": {"name": tool.name}},
        )
        prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
        ai_msg = model_with_tool_choice.invoke(prompt)
        if self.verbose:
            response_time = time.perf_counter() - start_time
            print(f"response finish in {response_time:.3f}sec")
        if ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            if self.verbose:
                print(f"tool_call: {tool_call['name']}, args: {tool_call['args']}")
            tool_result = tool.invoke(tool_call["args"])
            if update_messages:
                messages.append(ai_msg)
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
                final_prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
            else:
                temp_messages = messages.copy()
                temp_messages.append(ai_msg)
                temp_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
                final_prompt = LangModel.get_chat_prompt_from_messages(messages=temp_messages)
            return self.invoke_from_prompt(prompt=final_prompt, verbose=self.verbose)
        else:
            return self.parse_ai_msg(ai_msg=ai_msg)

    async def chat_astream(
        self,
        system_msg: str,
        human_msg: str,
        conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        print_processing_time: bool = False,
    ) -> AsyncIterator[tuple[str | list, dict, dict]]:  # [content, response_metadata, usage_metadata]
        messages = LangModel.get_messages(
            system_msg=system_msg,
            human_msg=human_msg,
            conversation_history=conversation_history,
        )
        return self.chat_from_messages_astream(messages=messages, print_processing_time=print_processing_time)

    async def chat_from_messages_astream(
        self,
        messages: list[BaseMessage],
        print_processing_time: bool = False,
    ) -> AsyncIterator[tuple[str | list, dict, dict]]:
        prompt = LangModel.get_chat_prompt_from_messages(messages=messages)
        if print_processing_time:
            start_time = time.perf_counter()
            first_token = True
            input_tokens = 0
            output_tokens = 0
        async for content, response_metadata, usage_metadata in self.astream_from_prompt(prompt=prompt):
            if print_processing_time and first_token and content:
                time_to_first_token = time.perf_counter() - start_time
                first_token = False
            if print_processing_time and usage_metadata and self.output_tokens_tag in usage_metadata:
                output_tokens = usage_metadata[self.output_tokens_tag]
            if print_processing_time and usage_metadata and self.input_tokens_tag in usage_metadata:
                input_tokens = usage_metadata[self.input_tokens_tag]
            yield (content, response_metadata, usage_metadata)
        if print_processing_time:
            time_to_last_token = time.perf_counter() - start_time
            tokens_per_second = (
                output_tokens / (time_to_last_token - time_to_first_token)
                if (time_to_last_token - time_to_first_token) > 0
                else 0
            )
            print(
                f"\nTTFT: {time_to_first_token:.3f} sec, TTLT: {time_to_last_token:.3f} sec, Prefill: {input_tokens / time_to_first_token:.3f} tps ({input_tokens} tokens / {time_to_first_token:.3f} sec), Decode: {tokens_per_second:.3f} tps ({output_tokens} tokens / {time_to_last_token - time_to_first_token:.3f} sec)"
            )

    async def chat_from_messages_astream_sentences(
        self,
        messages: list[BaseMessage],
        max_chars: int | None = None,
        print_processing_time: bool = False,
    ) -> AsyncIterator[tuple[str | list, dict, dict]]:
        buffer = AIMessageBuffer(max_chars=max_chars)
        async for content, response_metadata, usage_metadata in self.chat_from_messages_astream(
            messages=messages, print_processing_time=print_processing_time
        ):
            buffer.append(content)
            while (sentence := buffer.pop()) is not None:
                yield (sentence, response_metadata, usage_metadata)
        if (remaining := buffer.flush()) is not None:
            yield (remaining, response_metadata, usage_metadata)

    @staticmethod
    def get_image_file_recognition_prompt(
        system_msg: str,
        human_msg: str,
        image_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        jpeg_quality: int = 80,
        tmp_image_dir: str = ".",
    ) -> PromptValue:
        if max_length is not None:
            query_image_files = resize_image_files(
                src_image_files=image_files,
                dst_image_dir=tmp_image_dir,
                max_length=max_length,
                jpeg_quality=jpeg_quality,
            )
        else:
            query_image_files = image_files
        system_message = SystemMessage(content=system_msg)
        human_prompt = "{user_input}"
        frames = [convert_image_to_base64(file) for file in query_image_files]
        human_message_template = HumanMessagePromptTemplate.from_template(
            [
                human_prompt,
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{x}"},
                    },
                    frames,
                ),
            ]
        )
        messages = [system_message]
        if prior_conversation_history is not None:
            for index, msg in enumerate(prior_conversation_history):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(AIMessage(content=msg))
        messages.append(human_message_template)
        if post_conversation_history is not None:
            for index, msg in enumerate(post_conversation_history):
                if index % 2 == 0:
                    messages.append(AIMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(HumanMessage(content=msg))
        # prompt = ChatPromptTemplate.from_messages([system_message, human_message_template])
        template = ChatPromptTemplate.from_messages(messages=messages)
        return template.invoke({"user_input": human_msg})

    @staticmethod
    def get_image_recognition_prompt(
        system_msg: str,
        human_msg: str,
        images: list[cv2.typing.MatLike],
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
        verbose: bool = False,
    ) -> PromptValue:
        if max_length is not None:
            query_images = resize_images(
                src_images=images, max_length=max_length, interpolation=interpolation, verbose=verbose
            )
        else:
            query_images = images
        system_message = SystemMessage(content=system_msg)
        human_prompt = "{user_input}"
        # frames = [convert_image_to_base64(file) for file in query_image_files]
        frames = [convert_matlike_to_base64(image) for image in query_images]
        human_message_template = HumanMessagePromptTemplate.from_template(
            [
                human_prompt,
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{x}"},
                    },
                    frames,
                ),
            ]
        )
        messages = [system_message]
        if prior_conversation_history is not None:
            for index, msg in enumerate(prior_conversation_history):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(AIMessage(content=msg))
        messages.append(human_message_template)
        if post_conversation_history is not None:
            for index, msg in enumerate(post_conversation_history):
                if index % 2 == 0:
                    messages.append(AIMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(HumanMessage(content=msg))
        # prompt = ChatPromptTemplate.from_messages([system_message, human_message_template])
        template = ChatPromptTemplate.from_messages(messages=messages)
        return template.invoke({"user_input": human_msg})

    @staticmethod
    def get_video_recognition_prompt(
        system_msg: str,
        human_msg: str,
        video_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
    ):
        system_message = SystemMessage(content=system_msg)
        human_prompt = "{user_input}"
        videos = [convert_mp4_to_base64(file) for file in video_files]
        human_message_template = HumanMessagePromptTemplate.from_template(
            [
                human_prompt,
                *map(
                    lambda x: {
                        "type": "media",
                        "mime_type": "video/mp4",
                        "data": x,
                    },
                    videos,
                ),
            ]
        )
        messages = [system_message]
        if prior_conversation_history is not None:
            for index, msg in enumerate(prior_conversation_history):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(AIMessage(content=msg))
        messages.append(human_message_template)
        if post_conversation_history is not None:
            for index, msg in enumerate(post_conversation_history):
                if index % 2 == 0:
                    messages.append(AIMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(HumanMessage(content=msg))
        template = ChatPromptTemplate.from_messages(messages=messages)
        return template.invoke({"user_input": human_msg})

    def get_image_recognition_chain(
        self,
        system_msg: str,
        image_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
    ):
        system_message = SystemMessage(content=system_msg)
        human_prompt = "{text}"
        frames = [convert_image_to_base64(file) for file in image_files]
        human_message_template = HumanMessagePromptTemplate.from_template(
            [
                human_prompt,
                *map(
                    lambda x: {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{x}"},
                    },
                    frames,
                ),
            ]
        )
        messages = [system_message]
        if prior_conversation_history is not None:
            for index, msg in enumerate(prior_conversation_history):
                if index % 2 == 0:
                    messages.append(HumanMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(AIMessage(content=msg))
        messages.append(human_message_template)
        if post_conversation_history is not None:
            for index, msg in enumerate(post_conversation_history):
                if index % 2 == 0:
                    messages.append(AIMessage(content=msg))
                elif index % 2 == 1:
                    messages.append(HumanMessage(content=msg))
        # prompt = ChatPromptTemplate.from_messages([system_message, human_message_template])
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        # print(prompt)
        chain = prompt | self._model
        return chain

    # def image_recognition(
    #     self,
    #     system_msg: str,
    #     human_msg: str,
    #     image_files: list,
    #     prior_conversation_history: list[str] = None,  # [human_msg1, ai_msg1, ...]
    #     post_conversation_history: list[str] = None,  # [ai_msg1, human_msg1, ...]
    #     max_length: int | None = None,
    #     jpeg_quality: int = 80,
    #     tmp_image_dir: str = ".",
    # ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
    #     prompt = LangModel.get_image_file_recognition_prompt(
    #         system_msg=system_msg,
    #         human_msg=human_msg,
    #         image_files=image_files,
    #         prior_conversation_history=prior_conversation_history,
    #         post_conversation_history=post_conversation_history,
    #         max_length=max_length,
    #         jpeg_quality=jpeg_quality,
    #         tmp_image_dir=tmp_image_dir,
    #     )
    #     return self.invoke_from_prompt(prompt=prompt)

    def image_recognition(
        self,
        system_msg: str,
        human_msg: str,
        images: list[cv2.typing.MatLike],
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
        verbose: bool = False,
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        prompt = LangModel.get_image_recognition_prompt(
            system_msg=system_msg,
            human_msg=human_msg,
            images=images,
            prior_conversation_history=prior_conversation_history,
            post_conversation_history=post_conversation_history,
            max_length=max_length,
            interpolation=interpolation,
            verbose=verbose,
        )
        return self.invoke_from_prompt(prompt=prompt, verbose=verbose)

    def image_file_recognition(
        self,
        system_msg: str,
        human_msg: str,
        image_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
        verbose: bool = False,
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        images = [cv2.imread(image_file) for image_file in image_files]
        return self.image_recognition(
            system_msg=system_msg,
            human_msg=human_msg,
            images=images,
            prior_conversation_history=prior_conversation_history,
            post_conversation_history=post_conversation_history,
            max_length=max_length,
            interpolation=interpolation,
            verbose=verbose,
        )

    async def image_recognition_astream(
        self,
        system_msg: str,
        human_msg: str,
        images: list[cv2.typing.MatLike],
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
        print_processing_time: bool = False,
    ) -> AsyncIterator[tuple[str, dict, dict]]:
        prompt = LangModel.get_image_recognition_prompt(
            system_msg=system_msg,
            human_msg=human_msg,
            images=images,
            prior_conversation_history=prior_conversation_history,
            post_conversation_history=post_conversation_history,
            max_length=max_length,
            interpolation=interpolation,
        )
        if print_processing_time:
            start_time = time.perf_counter()
            first_token = True
            input_tokens = 0
            output_tokens = 0
        async for content, response_metadata, usage_metadata in self.astream_from_prompt(prompt=prompt):
            if print_processing_time and first_token and content:
                time_to_first_token = time.perf_counter() - start_time
                first_token = False
            if print_processing_time and usage_metadata and self.output_tokens_tag in usage_metadata:
                output_tokens = usage_metadata[self.output_tokens_tag]
            if print_processing_time and usage_metadata and self.input_tokens_tag in usage_metadata:
                input_tokens = usage_metadata[self.input_tokens_tag]
            yield (content, response_metadata, usage_metadata)
        if print_processing_time:
            time_to_last_token = time.perf_counter() - start_time
            tokens_per_second = (
                output_tokens / (time_to_last_token - time_to_first_token)
                if (time_to_last_token - time_to_first_token) > 0
                else 0
            )
            print(
                f"\nTTFT: {time_to_first_token:.3f} sec, TTLT: {time_to_last_token:.3f} sec, Prefill: {input_tokens / time_to_first_token:.3f} tps ({input_tokens} tokens / {time_to_first_token:.3f} sec), Decode: {tokens_per_second:.3f} tps ({output_tokens} tokens / {time_to_last_token - time_to_first_token:.3f} sec)"
            )

    async def image_file_recognition_astream(
        self,
        system_msg: str,
        human_msg: str,
        image_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        max_length: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
        print_processing_time: bool = False,
    ) -> AsyncIterator[tuple[str, dict, dict]]:
        images = [cv2.imread(image_file) for image_file in image_files]
        async for content, response_metadata, usage_metadata in self.image_recognition_astream(
            system_msg=system_msg,
            human_msg=human_msg,
            images=images,
            prior_conversation_history=prior_conversation_history,
            post_conversation_history=post_conversation_history,
            max_length=max_length,
            interpolation=interpolation,
            print_processing_time=print_processing_time,
        ):
            yield (content, response_metadata, usage_metadata)

    def video_recognition(
        self,
        system_msg: str,
        human_msg: str,
        video_files: list,
        prior_conversation_history: list[str] | None = None,  # [human_msg1, ai_msg1, ...]
        post_conversation_history: list[str] | None = None,  # [ai_msg1, human_msg1, ...]
        tmp_image_dir: str = ".",
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]

        prompt = LangModel.get_video_recognition_prompt(
            system_msg=system_msg,
            human_msg=human_msg,
            video_files=video_files,
            prior_conversation_history=prior_conversation_history,
            post_conversation_history=post_conversation_history,
        )
        return self.invoke_from_prompt(prompt=prompt)

    def few_shot_image_recognition(
        self,
        system_msg: str,
        first_human_msg: str,
        few_shot_examples: tuple[str, str],  # (image_file, answer)
        final_human_msg: str,
        query_image_file: str,
        max_length: int | None = None,
        jpeg_quality: int = 80,
        tmp_image_dir: str = ".",
    ) -> tuple[str | list, dict, dict]:  # [content, response_metadata, usage_metadata]
        system_message = SystemMessage(content=system_msg)
        human_messages = []
        human_messages.append({"type": "text", "text": first_human_msg})
        if max_length is not None:
            for example_image_file, answer in few_shot_examples:
                resized_example_image_file = os.path.join(tmp_image_dir, os.path.basename(example_image_file))
                if os.path.splitext(resized_example_image_file)[1] == ".png":
                    resized_example_image_file = os.path.splitext(resized_example_image_file)[0] + ".jpg"
                resize_image_file(
                    src_image_path=example_image_file,
                    dst_image_path=resized_example_image_file,
                    max_length=max_length,
                    jpeg_quality=jpeg_quality,
                )
                human_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{convert_image_to_base64(resized_example_image_file)}"
                        },
                    }
                )
                human_messages.append(
                    {
                        "type": "text",
                        "text": answer,
                    }
                )
            human_messages.append({"type": "text", "text": final_human_msg})
            resized_query_image_file = os.path.join(tmp_image_dir, os.path.basename(query_image_file))
            if os.path.splitext(resized_query_image_file)[1] == ".png":
                resized_query_image_file = os.path.splitext(resized_query_image_file)[0] + ".jpg"
            resize_image_file(
                src_image_path=query_image_file,
                dst_image_path=resized_query_image_file,
                max_length=max_length,
                jpeg_quality=jpeg_quality,
            )
            human_messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{convert_image_to_base64(resized_query_image_file)}"},
                }
            )
        else:
            for example_image_file, answer in few_shot_examples:
                human_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{convert_image_to_base64(example_image_file)}"},
                    }
                )
                human_messages.append(
                    {
                        "type": "text",
                        "text": answer,
                    }
                )
            human_messages.append({"type": "text", "text": final_human_msg})
            human_messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{convert_image_to_base64(query_image_file)}"},
                }
            )
        human_message = HumanMessage(content=human_messages)
        prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        chain = prompt | self._model
        ai_msg = chain.invoke({})
        # return ai_msg.content
        if (
            isinstance(ai_msg.content, list)
            and isinstance(ai_msg.content[0], dict)
            and "type" in ai_msg.content[0]
            and ai_msg.content[0]["type"] in ai_msg.content[0]
        ):
            return (ai_msg.content[0][ai_msg.content[0]["type"]], ai_msg.response_metadata, ai_msg.usage_metadata)
        else:
            return (ai_msg.content, ai_msg.response_metadata, ai_msg.usage_metadata)

    async def async_image_recognition(
        self,
        system_msg: str,
        human_msg: str,
        image_files: list,
        max_len_str: int = -1,
        add_timestamp: bool = False,
    ) -> str:
        ai_msg = ""
        if add_timestamp:
            start_time = time.perf_counter()
        async for chunk in self.astream_image_recognition(
            system_msg=system_msg,
            human_msg=human_msg,
            image_files=image_files,
        ):
            ai_msg += chunk
            if len(ai_msg) >= max_len_str > 0:
                if add_timestamp:
                    responce_time = time.perf_counter() - start_time
                    ai_msg += f"({responce_time:.3f} sec)"
                return ai_msg
        if add_timestamp:
            responce_time = time.perf_counter() - start_time
            ai_msg += f"({responce_time:.3f} sec)"
        return ai_msg

    async def astream_image_recognition(
        self,
        system_msg: str,
        human_msg: str,
        image_files: list,
    ) -> AsyncIterator[str]:
        chain = self.get_image_recognition_chain(
            system_msg=system_msg,
            image_files=image_files,
        )
        async for chunk in chain.astream({"type": "text", "text": human_msg}):
            yield chunk.content

    async def image_recognition_parallel(
        self,
        system_msgs: list[str],
        human_msgs: list[str],
        image_files: list[list[str]],
        max_len_str: int = -1,
        add_timestamp: bool = False,
    ) -> list[str]:
        prompts = list(zip(system_msgs, human_msgs, image_files))
        tasks = [
            self.async_image_recognition(
                system_msg=system_msg,
                human_msg=human_msg,
                image_files=images,
                max_len_str=max_len_str,
                add_timestamp=add_timestamp,
            )
            for system_msg, human_msg, images in prompts
        ]
        ai_msgs = await asyncio.gather(*tasks)
        return ai_msgs
