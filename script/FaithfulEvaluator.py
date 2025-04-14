from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage
from autogen_core import Agent, CancellationToken
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Sequence
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage, UserMessage
import pandas as pd
import re

class FaithfulEvaluatorAgent(BaseChatAgent):
    def __init__(
        self, 
        name: str, 
        # model:str=["gpt-4o"], #to be modified, use or not?
        # api_key:str=[dict["API_keys"]["OpenAI"]], #to be modified, use or not?
        system_message: str = "You are a critic agent that evaluates narrative faithfulness.",
        description: str = "An agent that extracts and validates narrative faithfulness.", #The description should
       # describe the agent's capabilities and how to interact with it.
        **kwargs
    ):
        super().__init__(name=name, description=description)
        self._model_context = UnboundedChatCompletionContext() # what is this? what's this for?
        self._system_message = system_message
        self.extractor = None
        self.generator = None
        self.last_narrative = None
        self.ds_info = None
        self.llm = None
        # self._model_client = genai.Client(api_key=api_key) #to be modified
        self.default_shap_df = None  # to store precomputed SHAP df

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return (TextMessage,)
        
    def set_models(self, extractor_class, generator_class, ds_info, llm):
        """
        Set up the extraction and generation models
        
        Args:
            extractor_class: The ExtractionModel class
            generator_class: The GenerationModel class
            ds_info: Dataset information
            llm: Language model client
        """
        self.ds_info = ds_info
        self.llm = llm
        self.extractor = extractor_class(ds_info=ds_info, llm=llm)
        self.generator = generator_class(ds_info=ds_info, llm=llm)
        return f"Models set: {extractor_class.__name__} and {generator_class.__name__}"

    @staticmethod
    def check_ones_and_none(df: pd.DataFrame, features_dict: dict) -> str:
        """
        Checks each row of df for:
        - Values not equal to 0 (treated as an error)
        - Values that are None/NaN (didn't extract any value)

        Integrates feature names from features_dict based on their 'rank'.
        If no issues found, returns a '100.0% correct' message.
        """
        # Build a mapping from row index (rank) to feature name
        rank_to_feature = {}
        for feat_name, feat_info in features_dict.items():
            rank = feat_info.get('rank')
            if rank is not None:
                rank_to_feature[rank] = feat_name

        messages = []
        found_issues = False

        for idx, row in df.iterrows():
            feature_name = rank_to_feature.get(idx, f"Unknown_{idx}")
            
            cols_with_one = []
            cols_with_none = []
            
            for col, val in row.items():
                if pd.isnull(val):
                    cols_with_none.append(col)
                elif val != 0:
                    cols_with_one.append(col)

            if cols_with_one:
                found_issues = True
                messages.append(f"Feature {feature_name} contains (an) errors in {cols_with_one} value.")
            
            if cols_with_none:
                found_issues = True
                messages.append(f"Feature {feature_name} failed to extract value in {cols_with_none}. ")

        if not found_issues:
            return "After checking, the narrative is 100% faithful to the SHAP table."
        
        return "\n".join(messages)

    def prepare_critic_input(
        self,
        narrative: list[str],
        shap_df: Optional[pd.DataFrame] = None,
        model=None,
        x=None,
        y=None,
        tree=True
    ):
        if not self.extractor or not self.generator:
            raise ValueError("Models not initialized. Call set_models first.")
        
        # Store the narrative for future reference
        self.last_narrative = narrative

        if shap_df is None:
                if self.default_shap_df is not None:
                    shap_df = self.default_shap_df
                elif model is not None and x is not None and y is not None:
                    self.generator.gen_variables(model, x, y, tree)
                    shap_df = self.generator.explanation_list[0].head(4)
                else:
                    raise ValueError("SHAP explanations not available. Provide shap_df or call gen_variables.")
                
        # 3) Generate extractions
        extraction = self.extractor.generate_extractions(narrative)

        # 6) Get differences (rank, sign, value, etc.)
        rank_diff, sign_diff, value_diff, real_rank, extracted_rank = self.extractor.get_diff(
            extraction[0],
            shap_df
        )

        # 7) Build the diff DataFrame
        df_diff = pd.DataFrame({
            "rank": rank_diff,
            "sign": sign_diff,
            "value": value_diff
        })

        # 8) Validate and return
        return self.check_ones_and_none(df_diff, extraction[0])

    async def on_messages(self, messages: List[ChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message
        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")
        return final_response

    async def on_messages_stream(
        self, messages: List[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        # Extract the latest narrative from the messages
        latest_narrative = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                latest_narrative = msg.content
                break

        if latest_narrative is None:
            critique = "No valid narrative found to critique."
        else:
            # Generate critique using our custom extraction/validation
            # This should produce the exact same output as your standalone code
            try:
                # Simply call prepare_critic_input with just the narrative
                # Don't regenerate SHAP explanations as they should already be generated
                critique = self.prepare_critic_input(narrative=[latest_narrative])
            except Exception as e:
                critique = f"Error during extraction and critique: {str(e)}"
        
        # Add messages to context for record keeping
        for msg in messages:
            if hasattr(msg, "content"):
                await self._model_context.add_message(UserMessage(content=msg.content, source=msg.source))
                
        # Add our response to the context
        await self._model_context.add_message(AssistantMessage(content=critique, source=self.name))
        
        # Create a minimal usage object
        usage = RequestUsage(
            prompt_tokens=0,
            completion_tokens=len(critique)
        )
        
        # Yield the final response with our custom critique
        yield Response(
            chat_message=TextMessage(content=critique, source=self.name, models_usage=usage),
            inner_messages=[],
        )
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent by clearing the model context."""
        await self._model_context.clear()

