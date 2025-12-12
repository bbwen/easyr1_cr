import datasets 
from datasets import load_dataset
import copy
from dataset_processing import process_dataset
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, AutoModelForSequenceClassification
from eval.eval_utils import hash_dataset
from eval.eval_args import GlobalArgs, LocalConfig
from eval.check_functions import confidence_verifier, llm_confidence_verifier 
import gc, os, re, math, json
from tqdm import tqdm
import torch
import numpy as np

# Global model cache to avoid reloading
_model_cache = {}

def get_model_and_processor(model_name):
    """Load model and processor, caching to avoid reloading"""
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _model_cache[model_name] = (model, processor)
    return _model_cache[model_name]

def clear_model_cache(model_name=None):
    """Clear model from cache to free memory"""
    global _model_cache
    if model_name:
        if model_name in _model_cache:
            del _model_cache[model_name]
    else:
        _model_cache = {}
    gc.collect()
    torch.cuda.empty_cache()

class GenerationOutput:
    """Mimics vLLM output structure for compatibility"""
    def __init__(self, text, logprobs=None):
        self.text = text
        self.logprobs = logprobs

class RequestOutput:
    """Mimics vLLM RequestOutput structure"""
    def __init__(self, outputs):
        self.outputs = outputs

def generate_with_transformers(model, processor, prompts, n=1, temperature=0.7, max_tokens=512, seed=None, return_logprobs=False):
    """
    Generate text using transformers, mimicking vLLM's output format.
    
    Args:
        model: The loaded model
        processor: The processor/tokenizer
        prompts: List of prompt strings (already formatted)
        n: Number of generations per prompt
        temperature: Sampling temperature
        max_tokens: Maximum new tokens to generate
        seed: Random seed
        return_logprobs: Whether to return log probabilities
    
    Returns:
        List of RequestOutput objects (mimics vLLM output)
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    outputs = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        # Process the prompt
        inputs = processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        generated_outputs = []
        for _ in range(n):
            with torch.no_grad():
                if return_logprobs:
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    output_ids = output.sequences
                    scores = output.scores  # tuple of logits for each generated token
                    
                    # Calculate logprobs
                    logprobs_list = []
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = output_ids[0, input_len:]
                    
                    for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
                        log_probs = torch.log_softmax(score[0], dim=-1)
                        token_logprob = log_probs[token_id].item()
                        token_text = processor.decode([token_id])
                        logprobs_list.append({
                            token_id.item(): type('LogProb', (), {
                                'logprob': token_logprob,
                                'decoded_token': token_text
                            })()
                        })
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                    )
                    logprobs_list = None
                
                # Decode only the generated part
                input_len = inputs['input_ids'].shape[1]
                output_text = processor.batch_decode(
                    output_ids[:, input_len:], 
                    skip_special_tokens=True
                )[0]
                
                generated_outputs.append(GenerationOutput(output_text, logprobs_list))
        
        outputs.append(RequestOutput(generated_outputs))
    
    return outputs

def main(global_args, local_configs):
    try: 
        dataset = datasets.load_from_disk("./" + global_args.dataset_name)
    except:
        dataset = load_dataset(global_args.dataset_name)
    dataset = dataset[global_args.split]
    dataset = dataset.map(lambda x: hash_dataset(x, global_args.hash_key))
    if global_args.sample_size is not None:
        dataset = dataset.select(range(global_args.sample_size))
    final_dataset = copy.deepcopy(dataset)

    all_metrics = {}
    updated = False 
    run_metrics = {} 
    
    try:
        existing_dataset = load_dataset(global_args.store_name, split=global_args.split)
        print(f"Found existing dataset {global_args.store_name} with {len(existing_dataset)} samples")
        final_dataset = copy.deepcopy(existing_dataset)
    except:
        try:
            existing_dataset = datasets.load_from_disk(global_args.store_name)
            print(f"Found existing dataset {global_args.store_name} with {len(existing_dataset)} samples")
            final_dataset = copy.deepcopy(existing_dataset)
        except:
            print(f"No existing dataset found for {global_args.store_name}")
            existing_dataset = None

    for config in local_configs:
        config.split = global_args.split
        if global_args.fresh:
            config.fresh = True
        out_dict = None
        available = False 
        run_metrics[config.name] = {}

        if existing_dataset is not None:
            if f"{config.name}-output_0" in existing_dataset.column_names:
                available = True
                if not config.fresh:
                    print(f"Skipping {config.name} because it already exists")
                    continue
                else:
                    updated = True 
                    print(f"Overwriting {config.name} because fresh is True")
        
        name = config.name
        config.dataset_name = global_args.dataset_name
        local_dataset = copy.deepcopy(dataset)
        local_dataset = process_dataset(local_dataset, config)

        ##### GENERATION #####
        
        model, processor = get_model_and_processor(config.model)
        
        # Get prompts
        to_tokenize = [local_dataset[i][config.tokenize_key] for i in range(len(local_dataset))]
        texts = processor.apply_chat_template(to_tokenize, add_generation_prompt=True, tokenize=False)
        
        # Handle if apply_chat_template returns a single string or list
        if isinstance(texts, str):
            texts = [texts]
        
        print("Prompt samples for config: ", name)
        print(texts[0])

        # Determine if we need logprobs
        need_logprobs = "confidence_prob" in config.vllm_task if hasattr(config, 'vllm_task') else False
        
        # Generate outputs
        outputs = generate_with_transformers(
            model, 
            processor, 
            texts, 
            n=config.n, 
            temperature=config.temperature, 
            max_tokens=config.max_tokens, 
            seed=config.seed,
            return_logprobs=need_logprobs
        )

        ##### POST-GENERATION PROCESSING #####

        if "ans_at_end" in config.vllm_task:
            inst = "Thinking time ended \n\n. My final answer is "
            prompts = []
            prompt_indices = []  # Track which output each prompt corresponds to
            
            for idx, (text, output) in enumerate(zip(texts, outputs)):
                for i in range(config.n):
                    prompts.append(text + output.outputs[i].text + inst)
                    prompt_indices.append((idx, i))
            
            ans_outputs = generate_with_transformers(
                model, processor, prompts, n=1, temperature=0, max_tokens=50
            )

            ans_calls_needed = 0 
            counter = 0 
            for out in outputs:
                for j in range(config.n):
                    ans_pattern = r"<answer>(.*?)</answer>"
                    ans_matches = re.findall(ans_pattern, out.outputs[j].text, re.DOTALL | re.MULTILINE)
                    last_answer = ans_matches[-1] if ans_matches else ""
                    if last_answer == "":
                        last_answer = ans_outputs[counter].outputs[0].text
                        out.outputs[j].text = out.outputs[j].text + "<answer> " + last_answer + " </answer>"
                        ans_calls_needed += 1
                    counter += 1 
            print(f"Number of answer calls needed for {config.name}: {ans_calls_needed/(config.n*len(outputs))}") 
            run_metrics[config.name]["ans_calls_needed"] = ans_calls_needed/(config.n*len(outputs)) 

        if "gen_then_classify" in config.vllm_task:
            clear_model_cache(config.model)

            ques_key = "problem" if "problem" in local_dataset.column_names else "question"

            if config.split_at_confidence:
                print(f"Splitting at confidence for {config.name}")
                for output in outputs:
                    output.outputs[0].text = output.outputs[0].text.split("<confidence>")[0]

            classify_texts = [
                f"\n\nPROBLEM: {local_dataset[i][ques_key]}\n\nEND OF PROBLEM\n\nMODEL'S RESPONSE: {output.outputs[0].text}\n\nEND OF RESPONSE\n\n" 
                for i, output in enumerate(outputs)
            ]
            
            print("Gen and Classify Samples for config: ", name)
            print(classify_texts[0])
            print(classify_texts[1])
            
            class_outputs = [] 
            if config.use_hf:
                class_model = AutoModelForSequenceClassification.from_pretrained(config.class_model).to("cuda")
                class_model.eval()
                class_tokenizer = AutoTokenizer.from_pretrained(config.class_model)
                
                batch_size = 16
                for i in tqdm(range(0, len(classify_texts), batch_size), desc="Classifying texts"):
                    batch_texts = classify_texts[i:i + batch_size] if i + batch_size <= len(classify_texts) else classify_texts[i:]
                    inputs = class_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
                    with torch.no_grad():
                        inputs = inputs.to("cuda")
                        output = class_model(**inputs)
                        output_tensor = output.logits.cpu().detach().numpy()
                        float_probs = output_tensor[:, 0]
                        fps = [] 
                        for fp in float_probs:
                            fps.append(1 / (1 + math.exp(-fp)))
                        class_outputs.extend(fps)
                
                del class_model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # For non-HF classifiers, you may need custom implementation
                raise NotImplementedError("Non-HF classifier not implemented in transformers version")

        if "confidence_prob" in config.vllm_task:
            invalid_count = 0 
            for output in outputs:
                for i in range(config.n):
                    picked = output.outputs[i]
                    if picked.logprobs is None:
                        output.outputs[i].text = output.outputs[i].text + f"<confidence> 0.5 </confidence>"
                        invalid_count += 1
                        continue
                        
                    len_gen = len(picked.logprobs)
                    tokens = [] 
                    probs = [] 
                    for j in range(len_gen):
                        lp_val = next(iter(picked.logprobs[j].values())).logprob
                        token = next(iter(picked.logprobs[j].values())).decoded_token 
                        probs.append(np.exp(lp_val))
                        tokens.append(token)
                    
                    answer_indices = [i for i, token in enumerate(tokens) if token == 'answer']
                    end_index = answer_indices[-1] if len(answer_indices) >= 1 else None
                    start_index = answer_indices[-2] if len(answer_indices) >= 2 else None
                    
                    if start_index is None or end_index is None or end_index - start_index >= 30:
                        output.outputs[i].text = output.outputs[i].text + f"<confidence> 0.5 </confidence>"
                        invalid_count += 1 
                    else:
                        selected_probs = probs[start_index:end_index]
                        avg_prob = sum(selected_probs) / len(selected_probs)
                        output.outputs[i].text = output.outputs[i].text + f"<confidence> {avg_prob} </confidence>"
                        
            print(f"Number of invalid confidence calls for {config.name}: {invalid_count/(config.n*len(outputs))}")
            run_metrics[config.name]["invalid_confidence_prob_calls"] = invalid_count/(config.n*len(outputs)) 

        if "confidence_at_end" in config.vllm_task: 
            inst = "Thinking time ended \n\n. My verbalized confidence in my answer as a number between 0 and 100 is equal to "
            prompts = []
            for text, output in zip(texts, outputs):
                for i in range(config.n):
                    prompts.append(text + output.outputs[i].text + inst)

            verb_outputs = generate_with_transformers(
                model, processor, prompts, n=1, temperature=0, max_tokens=20
            )

            conf_calls_needed = 0
            counter = 0 
            for output in outputs:
                for i in range(config.n):
                    conf_pattern = r"<confidence>(.*?)</confidence>"
                    conf_matches = re.findall(conf_pattern, output.outputs[i].text, re.DOTALL | re.MULTILINE)
                    last_confidence = conf_matches[-1] if conf_matches else ""
                    if last_confidence == "":
                        last_confidence = verb_outputs[counter].outputs[0].text
                        output.outputs[i].text = output.outputs[i].text + "<confidence>" + last_confidence + "</confidence>"
                        conf_calls_needed += 1
                    counter += 1 
            print(f"Number of confidence calls needed for {config.name}: {conf_calls_needed/(config.n*len(outputs))}")
            run_metrics[config.name]["conf_calls_needed"] = conf_calls_needed/(config.n*len(outputs)) 

        if out_dict is None:
            out_dict = {}
            for i in range(config.n):
                out_dict[f"{name}-output_{i}"] = []
            for output in outputs:
                for i in range(len(output.outputs)):
                    out_dict[f"{name}-output_{i}"].append(output.outputs[i].text)

        if "gen_then_classify" in config.vllm_task:
            out_dict[f"{name}-class_output"] = [] 
            for output in class_outputs:
                if config.use_hf:
                    out_dict[f"{name}-class_output"].append(output)
                else:
                    out_dict[f"{name}-class_output"].append(output.outputs.probs)
        
        for k, v in out_dict.items():
            if available:
                final_dataset = final_dataset.remove_columns([k]) 
            final_dataset = final_dataset.add_column(k, v)
            local_dataset = local_dataset.add_column(k, v)

        # Clear model cache between configs if needed
        # clear_model_cache(config.model)

        ##### CHECK FUNCTION #####
        
        if config.check_fn is not None:
            check_fn = config.check_fn
            if check_fn == "confidence_verifier":
                label_dict, metrics = confidence_verifier(local_dataset, config, **config.check_fn_args)
            elif check_fn == "llm_confidence_verifier":
                label_dict, metrics = llm_confidence_verifier(local_dataset, config, **config.check_fn_args)
            
            all_metrics[config.name] = metrics
            for k, v in label_dict.items():
                if available:
                    final_dataset = final_dataset.remove_columns([k]) 
                final_dataset = final_dataset.add_column(k, v)
                local_dataset = local_dataset.add_column(k, v)

    ##### END OF FOR LOOP AND CONFIG EVALUATION #####

    ##### PRINT ALL METRICS and LOG #####
        
    for config_name, metrics in all_metrics.items():
        print(f"Metrics for {config_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    for config_name, metrics in run_metrics.items():
        print(f"Run metrics for {config_name}:")
        try:
            for k, v in metrics.items():
                print(f"{k}: {v}")
        except:
            pass

    if global_args.log_path is not None:
        if not os.path.exists(global_args.log_path):
            os.makedirs(global_args.log_path)
        with open(global_args.log_path + "/metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)

    if updated or existing_dataset is None:
        final_dataset.save_to_disk(global_args.store_name)
        print(f"Dataset saved to {global_args.store_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="The name of the config to use")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    local_configs = []
    for i, c in enumerate(config):
        if i == 0:
            global_args = GlobalArgs(**c)
        else:
            local_configs.append(LocalConfig(**c))
     
    main(global_args, local_configs)