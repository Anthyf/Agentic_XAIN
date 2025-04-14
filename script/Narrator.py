from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, AgentEvent, TextMessage
from typing import AsyncGenerator, List

import re

class NarratorAgent(AssistantAgent):
    async def on_messages_stream(
        self,
        messages: List[ChatMessage],
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:

        # Let AssistantAgent do its normal work
        async for event in super().on_messages_stream(messages, cancellation_token):
            if isinstance(event, Response) and isinstance(event.chat_message, TextMessage):
                # 💡 Reformat the final reply before sending
                formatted_text = self.format_to_line_separated_sentences(event.chat_message.content)
                event.chat_message.content = formatted_text
            yield event

    @staticmethod
    def format_to_line_separated_sentences(text: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        return "\n".join(sentences)
