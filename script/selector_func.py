
from autogen_agentchat.messages import AgentEvent, ChatMessage
from typing import Sequence
import re

faithfulness_reached = False
coherence_phase_started = False

def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
    """
    This function defines how multiple agents work together.
    Initially, the narrator works with faithful_evaluator and faithful_critic iteratively in order to get a fully failthful narrative. 
    After the situation is satisfied, the cohereceagent first start to work. 
    In the new iteration, it only works with narrator until the narrative reaches 100% coherent.
    """
    global faithfulness_reached, coherence_phase_started

    # If no messages yet, start with narrator
    if len(messages) == 1:  # Only the initial prompt
        return "narrator"
    
    # Get the last message
    last_message = messages[-1]
    
    # PHASE 1: Faithfulness Iteration between narrator, extractor, and faithful agent
    if not faithfulness_reached:
        # Check if extractor has indicated 100% faithfulness
        if last_message.source == "faithful_evaluator":
            if "After checking, the narrative is 100% faithful to the SHAP table." in last_message.content:
                faithfulness_reached = True
                coherence_phase_started = True
                
                # Extract the latest narrator's message before activating coherenceagent
                latest_narrative = ""
                for msg in reversed(messages):
                    if msg.source == "narrator":
                        latest_narrative = msg.content
                        break
                
                # Transition to coherence agent
                return "coherenceagent"
        
        # Standard faithfulness cycle
        if last_message.source == "narrator":
            return "faithful_evaluator"
        elif last_message.source == "faithful_evaluator":
            return "faithful_critic"
        elif last_message.source == "faithful_critic":
            return "narrator"  # Back to narrator for refinement
    
    # PHASE 2: Coherence Iteration (only narrator and coherenceagent)
    if coherence_phase_started:
        if last_message.source == "narrator":
            return "coherenceagent"
        elif last_message.source == "coherenceagent":
            return "narrator"  # Continue the coherence improvement cycle
    
    # Default fallback (should not reach here)
    return "narrator"  # Default to narrator
