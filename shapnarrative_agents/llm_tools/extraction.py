import ast
import logging
import re
from typing import Type, Tuple
import pandas as pd
import shap
import numpy as np
from shapnarrative_agents.llm_tools.llm_wrappers import LLMWrapper
import json

class ExtractionModel:
    """
    A class to extract features and their importance from generated SHAP story narratives and then assign scores

    Attributes:
    -----------
    ds_info: dictionary
        Dictionary that contains short generic information about the dataset  
        Format: {
                 "dataset description": dataset_description (str), 
                 "target_description": target_description (str), 
                 "task_description": task_description (str), 
                 "feature_desc": pd.DataFrame that contains columns "feature_name", "feature_average", "feature_desc".
                 }
    llm : str, default="None"
        LLM API wrapper to do the extractions.
    """

    def __init__(
        self,
        ds_info: dict,
        llm: Type[LLMWrapper] = None,
    ):

        self.ds_info=ds_info
        self.feature_desc = ds_info["feature_description"]
        self.llm = llm

    def generate_prompt(self, narrative: str):

        """
        Generates SHAPstories for each instance in the given data.

        Parameters:
        -----------
        narrative : str
            A SHAP narrative that was generated to explain the prediction of a particular instance. 

        Returns:
        --------
        prompt_string: str
            A prompt for the extractor model 
        """

        prompt_string = f"""
        ***Context***:
        An LLM was used to create a narrative to explain and interpret a prediction made by another smaller classifier model. 
        The LLM was given an explanation of the classifier task, the training data, and provided with the exact names of all the features and their meaning. 
        Most importantly, the LLM was provided with a table that contains the feature values of that particular instance, the average feature values and their SHAP values which are a numeric measure of their importance. 
        
        You are an helpful agent tasked with improving the narrative. 
        To do so, you should extract some information about all the features that were mentioned in the narrative that will be given below.

        Here is some general info for you:
        Dataset description: {self.ds_info["dataset_description"]}.
        Target description: {self.ds_info["target_description"]}.
        Task description: {self.ds_info["task_description"]}.
        Feature descriptions: {self.feature_desc[["feature_name","feature_desc"]].to_string(index = False)}.
        
        ***Input text***: 
        The following is the given narrative.
        ====================
        \"\"\"{narrative}\"\"\"
        ====================

        ***Output Structure***:
        Provide your answer as a python dictionary with the keys as the feature names. 
        The values corresponding to the feature name keys are dictionaries themselves that contain the following inner keys: 
        1) "rank:" indicating the order of absolute importance of the feature starting from 0.
        2) "sign": the sign of whether the feature contributed towards target value 1 or against it (either +1 or -1 for sign value).
        3) "value": if the value of the feature is mentioned in a way that you can put an exact number on, add it. Only return numeric values here.
        If the description of the value is qualitative such as "many" or "often" and not mentioning an exact value, return "None" for its value.
        4) "assumption": give a short but complete 1 sentence summary of what the assumption is in the story for this feature.
        Provide this assumption as a general statement that could be fact checked later and that does not require this narrative as context.
        If no reason or suggestion is made in the story do not make something up and just return string 'None'.
        
        ***Guidelines***:
        1) Make sure that both the "rank", "sign", "value" and "assumption" keys and their values are always present in the inner dictionaries.
        2) Make sure that the "rank" key is sorted from 0 to an increasing value in the dictionary. The first element cannot have any other rank than 0.
        3) Make sure to use the exact names of the features as provided in the Feature descriptions, including capitalization. 
        4) Just provide the python dictionary as a string and add nothing else to the answer.
        """

        return prompt_string
    
    # @staticmethod
    # def extract_dict_from_str(extracted_str: str)->dict:

    #     """
    #     Extracts a dictionary from a string 

    #     Parameters:
    #     -----------
    #     extracted_str : str
    #         The answer to the extractor prompt, usually a simple dictionary in string form but could be preceded by some sentences.

    #     Returns:
    #     --------
    #     extracted_dict: dict
    #         A prompt for the extractor model 
    #     """

    #     start_index = extracted_str.find("{")
    #     end_index = extracted_str.rfind("}")
    #     dict_str = extracted_str[start_index : end_index + 1]

    #     extracted_dict = ast.literal_eval(dict_str)

    #     return extracted_dict

    

    @staticmethod
    def extract_dict_from_str(extracted_str: str) -> dict:
        """
        Extracts a dictionary from an LLM answer string, repairing common formatting errors.

        Parameters
        ----------
        extracted_str : str
            The LLM answer, usually a dict in string form, possibly surrounded by text.

        Returns
        -------
        dict
            The parsed dictionary, or {} if every parsing attempt fails.
        """

        logger = logging.getLogger(__name__)
        # keys whose quotes the LLM sometimes drops; edit to match your extraction schema
        KNOWN_KEYS = ("rank", "sign", "value", "assumption")
        
        if not extracted_str or not extracted_str.strip():
            logger.error("LLM output is empty or blank.")
            return {}

        extracted_str = extracted_str.strip()
        
        start_index = extracted_str.find("{")
        end_index = extracted_str.rfind("}")
        if start_index == -1 or end_index == -1:
            logger.error("No dictionary found in LLM output:\n%s", extracted_str)
            return {}

        dict_str = extracted_str[start_index:end_index + 1].strip()

        def sanitize(s: str, aggressive: bool = False) -> str:
            # trailing commas before } or ]
            s = re.sub(r",\s*([}\]])", r"\1", s)
            # missing values: '"value": ,' or '"rank": }'
            s = re.sub(r":\s*(?=[,}])", ": None", s)
            # JSON literals -> Python literals (only in value position, so text inside strings is untouched)
            s = re.sub(r"(:\s*)null\b", r"\1None", s)
            s = re.sub(r"(:\s*)true\b", r"\1True", s)
            s = re.sub(r"(:\s*)false\b", r"\1False", s)
            # missing comma between entries:  } "key"  ->  }, "key"
            s = re.sub(r"\}\s*(?=\")", "}, ", s)
            # truncated output: balance braces
            if s.count("{") > s.count("}"):
                s += "}" * (s.count("{") - s.count("}"))
            # known keys with missing quotes
            keys = "|".join(KNOWN_KEYS)
            s = re.sub(rf'(?<!")\b({keys})\b"', r'"\1"', s)           # rank"  -> "rank"
            s = re.sub(rf'"\b({keys})\b(?!")', r'"\1"', s)            # "rank  -> "rank"
            s = re.sub(rf"\b({keys})\b(?=\s*:)", r'"\1"', s)          # rank:  -> "rank":
            if aggressive:
                # bare words as values (unknown, N/A, NaN, inf, ...) -> None
                s = re.sub(
                    r"(:\s*)(?!None\b|True\b|False\b)[A-Za-z_][\w/]*(?=\s*[,}])",
                    r"\1None",
                    s,
                )
            return s.strip(" \n\t;")

        attempts = (
            ("raw",        lambda: ast.literal_eval(dict_str)),
            ("json",       lambda: json.loads(dict_str)),
            ("sanitized",  lambda: ast.literal_eval(sanitize(dict_str))),
            ("aggressive", lambda: ast.literal_eval(sanitize(dict_str, aggressive=True))),
        )

        last_error = None
        for name, attempt in attempts:
            try:
                result = attempt()
                if isinstance(result, dict):
                    if name != "raw":
                        logger.warning("Parsed LLM dict using '%s' fallback.", name)
                    return result
            except (SyntaxError, ValueError) as e:  # json.JSONDecodeError is a ValueError
                last_error = e

        logger.error(
            "Failed to parse LLM dictionary (%s: %s). Returning empty dict.\nRaw string:\n%s",
            type(last_error).__name__, last_error, dict_str,
        )
        return {}

    @staticmethod
    def get_diff(extracted_dict: dict, explanation: pd.DataFrame):

        """
        Compares the extracted dict with the actual explanation and calculates their difference

        Parameters:
        -----------
        extracted_dict : str
            The dictionary extracted from the LLM answer.
        explanation: pd.DataFrame
            A dataframe containing a column with the SHAP values and feature values.

        Returns: Tuple[5x list] 
        --------
        """
        #### This calculation is a bit subtle because you can in principle have various types of hallucinations in the extracted dict.
        
        ###STEP1: WE COMPUTE DIFFERENCE FOR ALL EXTRACTED FEATURES THAT ACTUALLY EXIST:

        #1)make sure the explanation is sorted by SHAP values (this should be already the case if generated with SHAPstory):

        explanation["abs_SHAP"] = explanation["SHAP_value"].abs()
        explanation = explanation.sort_values(by="abs_SHAP", ascending=False)
        explanation.drop(columns=["abs_SHAP"])

        #2) create a dataframe out of the extracted dict
        df_extracted=pd.DataFrame(extracted_dict).T
        df_extracted.reset_index(inplace=True)
        df_extracted.rename(columns={"index":"feature_name"},inplace=True)

        #3) filter the real explanation on the features that were present in the extraction dict 
        cat_dtype = pd.CategoricalDtype(df_extracted["feature_name"], ordered=True)
        explanation['feature_name']=explanation['feature_name'].astype(cat_dtype)
        df_real = explanation[explanation.feature_name.isin(df_extracted["feature_name"])].sort_values(by="feature_name")
        
        #4) get a list of feature names that have been extracted but do not exist (usually doesn't happen but good check)
        incorrect_features = df_extracted[~df_extracted['feature_name'].isin(df_real['feature_name'])]['feature_name'] 
            
        #5) now that we have a separate list of the hallucinated features, continue only with the overlap of existing features
        df_extracted=df_extracted[df_extracted['feature_name'].isin(df_real['feature_name'])] 
        sign_series=df_real["SHAP_value"].map(lambda x: int(np.sign(x)))
        df_real.insert(1,"sign",sign_series)
        df_real.insert(1,"rank", df_real.index)
        df_real=df_real.drop(columns=["SHAP_value","feature_desc"])
        
        #6) for all the real features replace any non-numeric extracted element with np.nan
        rank_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["rank"].to_numpy()])
        sign_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["sign"].to_numpy()])
        value_array=np.array([np.nan if type(x) not in [np.float64, np.int64,np.float32, np.int32, int] else x for x in df_extracted["value"].to_numpy()])

        #7) compute the difference arrays that we intend to output
        rank_diff=(rank_array-df_real["rank"].to_numpy()).astype(float)
        sign_diff=(sign_array*df_real["sign"].to_numpy()<=0).astype(float) #If extracted sign × real sign ≤ 0 → different or missing sign → 1.0 (mismatch). If both have same sign → 0.0 (match).
        value_diff=(value_array-df_real["feature_value"].to_numpy()).astype(float)

        #also useful to get actual real rank and extracted rank lists
        real_rank=df_real["rank"].to_numpy().astype(int)
        extracted_rank = df_extracted["rank"].astype("Int64").to_numpy()

        
        ###STEP 2: Now account for the fact that we ignored hallucinated features previously, and add a np.inf for the difference there.  
        for idx in sorted(incorrect_features.index.sort_values()):

            print("""*** Warning: Some features extracted by model were not in the real feature list ***.
            If this warning is encountered too often this could be a sign that something is wrong.""")

            
            if idx >= len(rank_diff):
                # Insert at the last position
                rank_diff = np.append(rank_diff, np.inf)
            else:
                # Insert at the specified index
                rank_diff = np.insert(rank_diff, idx, np.inf)

            if idx>=len(sign_diff):
                sign_diff = np.append(sign_diff, np.inf)
            else:
                sign_diff = np.insert(sign_diff, idx, np.inf)
        
            if idx>=len(value_diff):
                value_diff = np.append(value_diff, np.inf)
            else:
                value_diff = np.insert(value_diff, idx, np.inf)

        # --- NEW ADDITION: Ignore very small differences (< 1.0) ---
        # when a narrative extracts the value correctly however an integer, there is a value difference less than 1, thus also outputing a difference. This wasn't what
        # we want, thus cleaning and ignoring the value difference when it is minor.
        value_diff_clean = []

        for diff in value_diff:
            if np.isfinite(diff) and abs(diff) < 1:
                value_diff_clean.append(0.0)
            else:
                value_diff_clean.append(diff)

        value_diff = np.array(value_diff_clean)
        # ------------------------------------------------------------

        ### So now at the end, the rank/sign/value-diff arrays contain the difference between the extracted feature and the real feature,
        ### and if the feature did not exist have an np.inf at that position, or if the extracted element was not numeric contain np.nan.

        return rank_diff.tolist() , sign_diff.tolist(), value_diff.tolist(), real_rank.tolist(), extracted_rank.tolist()
        
    def generate_extractions(self, narratives: list[str]):

        """
        Generates SHAPstories for each instance in the given data.

        Args:
            narratives: (list of strings) all the narratives that we want to extract the features of

        Returns:
            extractions (list of dicts): A list containing the extracted feature dicts for each narrative
        """

        extractions=[]
        for i, narrative in enumerate(narratives):
            extractions.append( 
            self.extract_dict_from_str(self.llm.generate_response(self.generate_prompt(narrative))) 
            )
            print(f"Extracted story {i+1}/{len(narratives)} with {self.llm.model}")


        return extractions