import pickle
import json
import itertools
import time
import pandas as pd
from typing import Type
from shapnarrative_agents.agents.prompt import Prompt
from shapnarrative_agents.agents.Narrator_Single import NarratorAgent
from shapnarrative_agents.agents.FaithfulEvaluator import FaithfulEvaluator
from shapnarrative_agents.llm_tools.llm_wrappers import LLMWrapper
from shapnarrative_agents.experiment_management.experiment_dataclass_agentic_coherence import AgenticNarrativeExperiment


class AgenticExperimentManager:
    
    """ 
    A class about the self-revision single-agent system.
    
    Attributes:
        dataset_names: (list) list of names for the target models to be used
        tar_model_names: (list) list of names for the target models to be used
        narrator_models: (list) list of narrator models
        faithfulevaluator_models: (list) list of faithfulevaluator models. Onlhy used for the final faithfulness metric evaluation, not for the feedback generation.
        prompt_types: (list) list of prompt types allowed in the generation model class
        ds_paths: (dict) dictionary with ds_names as keys, and the paths to that dataset location as values
        size_lim: (int) number of narratives to create for a given dataset
        num_feat: (int) number of features to use in t
    """
        
    def __init__(
        self,
        dataset_names: list[str],
        tar_model_names: list[str],
        narrator_models: list[Type[LLMWrapper]],
        faithfulevaluator_models: list[Type[LLMWrapper]],
        prompt_types: list[str],
        ds_paths: dict,
        size_lim: int,
        num_feat: int,
        temperature: float = 0.0,
        baseline_path: str = "data/baseline_narratives.json",
    ):   
        
        self.model_pairs = list(zip(narrator_models, faithfulevaluator_models))  
        self.dataset_names = dataset_names
        self.tar_model_names = tar_model_names
        self.narrator_models=narrator_models
        self.faithfulevaluator_models=faithfulevaluator_models
        self.prompt_types = prompt_types
        self.ds_paths = ds_paths
        self.size_limit = size_lim
        self.num_feat = num_feat
        self.temperature = temperature
        self.baseline_path = baseline_path
        self.baseline_lookup = self.load_narratives_json()

    def sample_equal_targets(self, df):

        """Given a dataframe representing a large test set randomly samples n_tot instances from it such that it has an equal number of classes
        Args:
            df: (pd.DataFrame) Test_set containing the class label in a "target" variable

        Returns:
            sampled_df: (pd.DataFrame) sampled test set to be used for narrative generation 
        """
        

        target_name=df.columns[-1]
        df_target_0 = df[df[target_name] == 0]
        df_target_1 = df[df[target_name] == 1]

        # Randomly sample entries from each subset
        sample_0 = df_target_0.sample(int(self.size_limit/2), random_state=42)
        sample_1 = df_target_1.sample(int(self.size_limit/2), random_state=42)

        # Combine the samples
        sampled_df = pd.concat([sample_0, sample_1])

        # Shuffle the combined sample to mix the rows
        sampled_df = sampled_df.sample(frac=1, random_state=42) 

        return sampled_df

    def dataset_extraction_tool(self, dataset_name, tar_model_name):

        """Given a dataset name reads in all necessary data from the data folder that we need for narrative generation

        Args:
            dataset_name: name of dataset to read the path through the self.ds_paths dictonary
            tar_model_name: name of the target model for which the SHAP values will be generated (e.g. RF)

        Returns:
            sampled_test_set: output of the sample_equal_targets method
            ds_info: dictionary that was prepared during data preprocessing containing necassary info for prompt
            tar_model: read-in pickled target model
        """
        
        test_set=pd.read_parquet(f"{self.ds_paths[dataset_name]}/test_cleaned.parquet") 
        sampled_test_set=self.sample_equal_targets(test_set)

        with open(f"{self.ds_paths[dataset_name]}/dataset_info", 'rb') as f:
            ds_info= pickle.load(f)
        
        with open(f"{self.ds_paths[dataset_name]}/{tar_model_name}.pkl", 'rb') as f:
            tar_model= pickle.load(f)

        return sampled_test_set, ds_info, tar_model

    def load_narratives_json(self):
        """Converts the JSON list of dicts into a fast lookup dictionary."""
        # Load Round 0 narratives from the JSON file that contains the baseline narratives
        with open(self.baseline_path, 'r', encoding='utf-8') as f:
            round0_data = json.load(f)

        lookup = {}
        for item in round0_data:
            key = (item['dataset'], item['index'])
            lookup[key] = item['round0_narrative']
        return lookup
        
    def get_baseline_narrative(self, dataset_name, idx):
        """Accesses the preloaded dictionary stored in lookup"""
        return self.baseline_lookup.get((dataset_name, idx), "No Round 0 narrative found.")
    

    def run_agentic_experiments(
         self,
         temp_save_path: str = "results/temp_agent/latest_experiment.pkl",
         warm_start: bool = False,
         manipulate: bool= False,
        ):
    
        """Executes experiments for this single agent system.
        args:
            temp_save_path: saves intermediate results in temporary dir, in case an api call fails to avoid redoing everything.
            warm_start: start from latest temp_save_path result?
            manipulate: bool -- manipulate narratives or not
        returns:
            list of NarrativeExperiment dataclasses as defined in the experiment_dataclass module.
        
        """
        experiments=[]
        
        
        #in case an api call failed in the previous experiment run, we can start from the last saved:
        if warm_start:
            with open(temp_save_path, 'rb') as f:
                experiments=pickle.load(f)
            existing_ids=[exp.id for exp in experiments]
        else:
            existing_ids = []
        
        # The total iterations should multiply the len of critic model and len of coherence model when they are different from narrator models;
        #here we didn't multiply them, because in this study we make all three models same.
        total_iterations = len(self.dataset_names) * len(self.tar_model_names) * len(self.narrator_models)* len(self.prompt_types)  
        
        # This study is for when we make narrator, evaluator, critic model, and coherence model 1 to 1 pairs; add faithfulcritic_model into the zip if use llm; add coherence_model in if it is present
        for index, (dataset_name, tar_model_name, (narrator_model, faithfulevaluator_model), prompt_type) in enumerate(
        itertools.product(self.dataset_names, self.tar_model_names, self.model_pairs, self.prompt_types)):
       
            exp_id = (dataset_name, tar_model_name, narrator_model.model, prompt_type)
            if exp_id in existing_ids:
                continue

            print(f"****** Running experiment {index+1}/{total_iterations} with {exp_id} ******")
            
            per_start_time = time.time()

            sampled_test_set, ds_info, tar_model = self.dataset_extraction_tool(dataset_name, tar_model_name)
            prompt_generator = Prompt(ds_info=ds_info)
            narrator = NarratorAgent(narrator_model)
            
            # This study:
            faithful_evaluator = FaithfulEvaluator(ds_info, llm=faithfulevaluator_model)

            idx_list = []
            explanation_list = []
            narratives_all = []
            results_list = []
            extractions_list = []        

            for idx in sampled_test_set.index:
                print(f"📌 Generating for instance index: {idx}")
                x = sampled_test_set[sampled_test_set.columns[0:-1]].loc[[idx]]
                y = sampled_test_set[sampled_test_set.columns[-1]].loc[[idx]]

                prompt_generator.gen_variables(tar_model, x, y, tree=True)
                shap_df = pd.DataFrame(prompt_generator.explanation_list[0])
                prompt = prompt_generator.generate_story_prompt(0, prompt_type=prompt_type, manipulate=manipulate) 

                round_narratives = []
                round_extractions = []

                # Round 0 narrative gets from the baseline narrative
                narrative = self.get_baseline_narrative(dataset_name,idx)
                print(f"📝 Baseline Narrative Round 0")
                round_narratives.append(narrative)

                for round_num in range(3):
                    extraction = faithful_evaluator.generate_extractions([[narrative]])  # expects list of lists
                    round_extractions.append(extraction)

                    rank_diff, sign_diff , value_diff, real_rank, extracted_rank=faithful_evaluator.get_diff(extraction[0],shap_df)
                            
                    df_diff = pd.DataFrame({
                        "rank": rank_diff,
                        "sign": sign_diff,
                        "value": value_diff
                    })

                    # this is only ensure the early stopping if the narrative is already 100% faithful, but the feedback won't be saved or use as input for narrator_single
                    faithful_feedback = faithful_evaluator.give_feedback(df_diff, extraction[0])
                    if "100% faithful" in faithful_feedback.lower():
                        print("✅ Feedback indicates 100% faithful narrative, stopping further rounds.")
                        break

                    narrative = narrator.generate(prompt, last_narrative=narrative)
                    print(f"📝 Narrative Round {round_num + 1}")
                    round_narratives.append(narrative)

                narratives_all.append(round_narratives)
                extractions_list.append(round_extractions)
                explanation_list.append(prompt_generator.explanation_list[0])
                results_list.append(prompt_generator.result_df)
                idx_list.append(idx)

            experiment = AgenticNarrativeExperiment(
                dataset=dataset_name,
                prompt_type=prompt_type,
                tar_model_name=tar_model_name,
                idx_list=idx_list,
                explanation_list=explanation_list,
                results_list=results_list,
                narrator_model=narrator_model.model,
                narrative_list=narratives_all,
                #This study: 
                extractions_dict={faithfulevaluator_model.model: extractions_list},
                id=exp_id,
                num_feat=self.num_feat
            )

            per_end_time = time.time()
            total_runtime = per_end_time - per_start_time
            print(f"\n✅ This experiment completed in {total_runtime:.2f} seconds "
                f"({total_runtime/60:.2f} minutes).")
        
            experiments.append(experiment)
            with open(temp_save_path, "wb") as f:
                pickle.dump(experiments, f)

        return experiments