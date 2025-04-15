from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import copy
from pydantic import RootModel


app = FastAPI()

class ExtractionBlock(BaseModel):
    assumption: Optional[str]

class ExtractionItem(RootModel):
    root: Dict[str, ExtractionBlock]

class MetricInput(BaseModel):
    dataset: str
    extractions_dict: Dict[str, List[Dict[str, ExtractionBlock]]]

class PerplexityRequest(BaseModel):
    metrics: List[MetricInput]
    models: List[str]
    hf_token: str

@app.post("/compute_perplexity/")
def compute_perplexity_endpoint(request: PerplexityRequest):

    # Convert incoming Pydantic models to Python dictionaries
    metrics = request.metrics
    models = request.models
    hf_token = request.hf_token

    results = []

    for model_name in models:
        # Load model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=hf_token
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        def perplexity(text):
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            return torch.exp(loss).item()

        # Loop through each metric dataset
        for metric in metrics:
            dataset = metric.dataset
            extract_dict = metric.extractions_dict

            ppl_dict = {}
            ppl_mean = {}

            for extr_model, items in extract_dict.items():
                model_ppls = []
                item_results = []

                for i, assumption_set in enumerate(items):
                    feature_ppls = {}
                    for feature, extr in assumption_set.items():
                        if extr.assumption in [None, "None"]:
                            print("⚠️  Skipping empty or None assumption")
                            ppl = None
                        else:
                            try:
                                ppl = perplexity(extr.assumption)
                                print(f"✅ Perplexity: {ppl}")
                            except Exception as e:
                                print(f"❌ Error computing perplexity: {e}")
                                ppl = None
                    # 🔧 These two lines were missing:
                    feature_ppls[feature] = ppl
                    if ppl is not None:
                        model_ppls.append(ppl)
                
                item_results.append(feature_ppls)  # <-- THIS was missing!

                ppl_dict[extr_model] = item_results
                ppl_mean[extr_model] = float(np.nanmean(model_ppls)) if model_ppls else None

            results.append({
                "dataset": dataset,
                "model_name": model_name,
                "perplexity_dict": ppl_dict,
                "perplexity_mean": ppl_mean
            })

    return {"status": "done", "results": results}
