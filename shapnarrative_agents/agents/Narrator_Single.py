from .LLMBaseAgent import LLMBaseAgent
from shapnarrative_agents.llm_tools.llm_wrappers import LLMWrapper
import re


class NarratorAgent(LLMBaseAgent):
    """
    This class defines the Narrator_Single agent that that revise its narrative based on self-reflection.
    In Round 0, the Narrator_Single only received the baseline narrative.
    In subsequent rounds, it revises its previous answer based on self-revision that compares with the SHAP table.
    """
    @staticmethod
    def format_to_line_separated_sentences(text: str) -> str:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        return "\n".join(sentences)
    
    def generate(self, initial_prompt, last_narrative=None):
        final_prompt = f"""
        ***Context***:
        You are a helpful agent who writes model explanations (narratives) based on SHAP values. 
        This is your initial task: {initial_prompt}.

        ***Input text***:
        ====================
        This is your previous narrative:{last_narrative}.
        ====================

        ***Output Structure***:
        The narrative MUST comply with all format related rules and content related rules from the initial task.

        ***Guidelines***: 
        1) You MUST check your narrative against the SHAP table given in the initial task and revise it if necessary.
        2) You MUST return the narrative only. DO NOT chitchat.
        """

        response = self.llm.generate_response(final_prompt)
        return self.format_to_line_separated_sentences(response)