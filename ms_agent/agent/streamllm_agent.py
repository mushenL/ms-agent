# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from copy import deepcopy
from typing import List, Union

import json
from ms_agent.llm.utils import Message
from ms_agent.utils.logger import logger

from .llm_agent import LLMAgent

class StreamLLMAgent(LLMAgent):
    """Agent running for llm-based tasks by stream.

    Args:
        config_dir_or_id (`Optional[str]`): The directory or id of the config file.
        config (`Optional[DictConfig]`): The configuration object.
        env (`Optional[Dict[str, str]]`): The extra environment variables.
        mcp_config(`Optional[Dict[str, Any]]`): The extra mcp config file location.
    """
    DEFAULT_SYSTEM = 'You are a helpful assistant.'

    async def _step(self, messages: List[Message], tag: str) -> List[Message]: # type: ignore
        messages = deepcopy(messages)
        # Refine memory
        messages = await self._refine_memory(messages)
        # Do plan
        messages = await self._update_plan(messages)
        await self._loop_callback('on_generate_response', messages)
        tools = await self.tool_manager.get_tools()
        if hasattr(self.config, 'generation_config') and getattr(
                self.config.generation_config, 'stream', False):
            self._log_output('[assistant]:', tag=tag)
            _content = ''
            is_first = True
            for _response_message in self._handle_stream_message(
                    messages, tools=tools):
                if is_first:
                    messages.append(_response_message)
                    is_first = False
                new_content = _response_message.content[len(_content):]
                sys.stdout.write(new_content)
                sys.stdout.flush()
                _content = _response_message.content
                yield messages
        else:
            _response_message = self.llm.generate(messages, tools=tools)
            if _response_message.content:
                self._log_output('[assistant]:', tag=tag)
                self._log_output(_response_message.content, tag=tag)
            messages.append(_response_message)
            yield messages
        if _response_message.tool_calls:
            self._log_output('[tool_calling]:', tag=tag)
            for tool_call in _response_message.tool_calls:
                tool_call = deepcopy(tool_call)
                if isinstance(tool_call['arguments'], str):
                    tool_call['arguments'] = json.loads(tool_call['arguments'])
                self._log_output(
                    json.dumps(tool_call, ensure_ascii=False, indent=4),
                    tag=tag)

        await self._loop_callback('after_generate_response', messages)
        await self._loop_callback('on_tool_call', messages)

        if _response_message.tool_calls:
            messages = await self._parallel_tool_call(messages)
        else:
            self.runtime.should_stop = True
        await self._loop_callback('after_tool_call', messages)
        yield messages
    
    async def run(self, inputs: Union[List[Message], str],
                  **kwargs):
        """Run the agent, mainly contains a llm calling and tool calling loop.

        Args:
            inputs(`Union[str, List[Message]]`): The inputs can be a prompt string,
                or a list of messages from the previous agent
        Returns:
            The final messages
        """
        try:
            self.max_chat_round = getattr(self.config, 'max_chat_round', 20)
            round = 0
            self._register_callback_from_config()
            self._prepare_llm()
            self._prepare_runtime()
            await self._prepare_tools()
            await self._prepare_memory()
            await self._prepare_planer()
            await self._prepare_rag()
            self.runtime.tag = self.tag
            messages = await self._prepare_messages(inputs)
            await self._loop_callback('on_task_begin', messages)
            if self.planer:
                messages = await self.planer.make_plan(self.runtime, messages)
            for message in messages:
                self._log_output('[' + message.role + ']:', tag=self.tag)
                self._log_output(message.content, tag=self.tag)
            while not self.runtime.should_stop:
                async for messages in self._step(messages, self.tag):
                    yield messages
                round += 1
                # +1 means the next round the assistant may give a conclusion
                if round >= self.max_chat_round + 1:
                    if not self.runtime.should_stop:
                        messages.append(
                            Message(
                                role='assistant',
                                content=f'Task {messages[1].content} failed, max round(20) exceeded.')
                        )
                    self.runtime.should_stop = True
                    yield messages
                    break
            await self._loop_callback('on_task_end', messages)
            await self._cleanup_tools()
        except Exception as e:
            if hasattr(self.config, 'help'):
                logger.error(
                    f'[{self.tag}] Runtime error, please follow the instructions:\n\n {self.config.help}'
                )
            raise e