import pandas as pd
import numpy as np
import yaml
import openai
import asyncio 
import shap
import pickle
from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

import sys
import os

sys.path.append("../SHAPnarrative-metrics")
sys.path.append("../script")

from shapnarrative_metrics.llm_tools import llm_wrappers
from shapnarrative_metrics.misc_tools.manipulations import full_inversion, shap_permutation
from shapnarrative_metrics.llm_tools.generation import GenerationModel
from shapnarrative_metrics.llm_tools.extraction import ExtractionModel
from ExtractorCritc import ExtractorCriticAgent

with open("../config/keys.yaml") as f:
    dict=yaml.safe_load(f)
api_key = dict["API_keys"]["OpenAI"]

dataset_name="student"

with open(f'../data/{dataset_name}_dataset/dataset_info', 'rb') as f:
   ds_info= pickle.load(f)

with open(f'../data/{dataset_name}_dataset/RF.pkl', 'rb') as f:
   trained_model=pickle.load(f)

train=pd.read_parquet(f"../data/{dataset_name}_dataset/train_cleaned.parquet")
test=pd.read_parquet(f"../data/{dataset_name}_dataset/test_cleaned.parquet")

print(test.index)

idx=309

x=test[test.columns[0:-1]].loc[[idx]]
y=test[test.columns[-1]].loc[[idx]]

# Generate SHAP table, prompt and story
TEMPERATURE=0
MANIP=True

gpt = llm_wrappers.GptApi(api_key, model="gpt-4o", system_role="You are a teacher that explains AI predictions.", temperature=TEMPERATURE)
generator=GenerationModel(ds_info=ds_info, llm=gpt)
generator.gen_variables(trained_model,x,y,tree=True)
shap_df=pd.DataFrame(generator.explanation_list[0].head(4))
prompt = generator.generate_story_prompt(0,prompt_type="long",manipulate=MANIP)
print(prompt)
story=generator.generate_stories(trained_model,x,y,prompt_type="long",manipulate=MANIP)

extractor=ExtractionModel(ds_info=ds_info, llm=gpt)
extraction=extractor.generate_extractions(story)
extraction
extraction_string=extractor.generate_prompt(story)
print(extraction_string)


rank_diff, sign_diff , value_diff, real_rank, extracted_rank=extractor.get_diff(extraction[0],generator.explanation_list[0])
#compare the difference between generated SHAP table from GenerationModel (explanation_list) vs extracted features from ExtractionModel (extraction[0])'''

df_diff = pd.DataFrame({
    "rank": rank_diff,
    "sign": sign_diff,
    "value": value_diff
})
print(df_diff)

feedback = ExtractorCriticAgent.check_ones_and_none(df_diff, extraction[0])
print(feedback)



prompt="""Your goal is to generate a textual explanation or narrative as to why an AI model made a certain prediction for one particular instance. 
        To do this, you will be provided with a dictionary that contains broad descriptions of the specific dataset, target variable, and the task the model was trained on.
        Additionally, you will be provided with a dataframe that contains the names of all the features, their descriptions, their values, their average values and their SHAP values.
        Finally you will get a single string describing the result of the prediction.

        The goal of SHAP is to explain the prediction of an instance by computing the contribution of each feature to the prediction.
        Each individual SHAP value is a measure of how much additional probability this feature adds or subtracts 
        in the predicted probability relative to the base level probability. 
        This relative nature of the SHAP values might have unexpected consequences that you are to take into account.
        For example, features that should intuitively contribute in a positive way (and vice versa), 
        can still have negative SHAP values if their value is below an average in the dataset.

        Here is the data:

        Dataset description: The dataset contains information about students from two Portugese high schools and in particular their family situation and other habits,
        Target description: The target variable represents the final year grade, transformed into whether the student passed (1) or not (0) at the end of the year,
        Task description": Predict whether a student will pass,
        Feature table: feature_name  SHAP_value  feature_value  feature_average                                                                                                                       feature_desc
        Fedu    0.030117              1         2.515823 Father's education (0: none, 1: primary education (4th grade), 2: 5th to 9th grade, 3: secondary education or 4: higher education)
   schoolsup    0.027889              1         0.139241                                                                         Student receiving extra educational support (0: no, 1:yes)
    failures    0.023675              1         0.360759                                                                                        Number of past class failures (from 0 to 3)
         age    0.022808             19        16.756329                                                                                                    The age of the student in years,
        Result: The model predicted a probability of 64.00% for the target variable being 1,
                    and therefore predicted the outcome 1.

        The features in the "Feature table" above are already sorted by their absolute SHAP values to make it easier for you to locate the most important ones. 
        Always provide the probability of getting outcome 1, even if the predicted outcome is 0. 

        Please generate a fluent and cohesive narrative that explains the prediction made by the model. In your answer follow these rules.

        Format related rules:

        1) You can start the answer immediately with the explanation. 
        2) The answer should end with a single sentence summarizing and weighing the different factors. 
        3) Limit the entire answer to 10 sentences or fewer. 
        4) Only invoke the 4 most important features in the narrative. 
        5) Do not use tables or lists, or simply rattle through the features one by one. The goal is to have a narrative/story.

        Content related rules:

        1) Be clear about what the model actually predicted and how certain it was about that prediction. 
        2) Discuss how the features contributed to that prediction and be clear about whether a particular feature had a positive or negative contribution relative to the probability of getting outcome 1.
           Make sure to clearly establish this the first time you refer to a feature.
        3) Consider the relative magnitude of the absolute SHAP values of the features when referencing their relative importance.
        4) The reader should be able to tell what the order of importance of the features is based on their absolute SHAP values. 
        5) You should provide a suggestion or interpretation as to why a feature contributed in a certain direction. Try to introduce external knowledge that you might have that is not in the SHAP table.  
        6) If there is no simple explanation for the effect of a feature, try to consider the context of other features in the interpretation.
        7) Do not use the SHAP numeric values in your answer.
        8) You can use the feature values themselves in the explanation, as long as they are not categorical variables.
        9) Do not use the comparison with the average for every single feature, and reserve it for features where it really clarifies the explanation."""

system_message_narrator = f"""
{prompt}
You should base your response on the feedback from the faithfulcritic agent to improve your answer.
"""

system_message_extractorcritic = """
You are an expert for SHAP narrative extraction analysis. 
Reply TERMINATE if you didn't find any errors and NaN after you extract the information."""


system_message_faithfulcritic = """
You should summarize the output information from extractorcritic and give your feedback to narrator.
If, according to extractorcritic, some features contain errors in ['rank'] value, please add an advice: Please consider change the importance order of the features.
Finish your feedback with a sentence: Please improve your narrative based on the given information.
"""

shap_df = shap_df

async def main() -> None:
    SEED = 42
    TEMPERATURE = 0

    # Initialize OpenAI client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=api_key,
        seed=SEED,
        temperature=TEMPERATURE
    )
    
    # Create the narrator agent
    narrator = AssistantAgent(
        name="narrator",
        system_message=system_message_narrator,
        model_client=model_client,
        reflect_on_tool_use=False,
    )
    
    # Create the critic agent with advanced extraction capabilities
    extractorcritic = ExtractorCriticAgent(
        name="extractorcritic",
        system_message=system_message_extractorcritic,
        description="An agent that extracts and validates narrative faithfulness",
    )
    
    # Set up the models for the critic - THIS IS IMPORTANT
    extractorcritic.set_models(
        extractor_class=ExtractionModel,
        generator_class=GenerationModel,
        ds_info=ds_info,
        llm=gpt
    )

    extractorcritic.default_shap_df = shap_df 

    faithfulcritic = AssistantAgent(
        name="faithfulcritic",
        system_message=system_message_faithfulcritic,
        model_client=model_client,
        reflect_on_tool_use=False,
    )

    # Create the group chat with custom processing
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)
    
    group_chat = RoundRobinGroupChat(
        [narrator, extractorcritic, faithfulcritic],
        termination_condition=termination
    )
    
    # Run the chat
    stream = group_chat.run_stream(task="start your narrative generation")
    
    await Console(stream)

if __name__ == "__main__":
    await main()