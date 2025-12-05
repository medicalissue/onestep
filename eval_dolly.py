import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="distill/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available. Using CPU.")
        device = torch.device("cpu")
        
    # Allow override from config if specified and valid
    if hasattr(cfg, 'distill') and hasattr(cfg.distill, 'solver_device'):
        if 'cuda' in cfg.distill.solver_device and not torch.cuda.is_available():
            print(f"Config requested {cfg.distill.solver_device} but CUDA is not available. Keeping {device}.")
        else:
            device = torch.device(cfg.distill.solver_device)
            
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    print(f"Loading Student Model: {cfg.model.student_name}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.student_name)
    
    # Load Weights
    # Load Weights
    # Priority: 1. Explicit path 2. Best model in output dir 3. Fallback
    
    if hasattr(cfg.model, 'checkpoint_path') and cfg.model.checkpoint_path:
        ckpt_path = cfg.model.checkpoint_path
    else:
        # Match train_llm.py output structure
        output_dir = f"outputs/{cfg.model.student_name}_{cfg.distill.method}_{cfg.wandb.name}"
        ckpt_path = os.path.join(output_dir, "best_model.pth")
        
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Using pretrained weights.")
        
    model.to(device)
    model.eval()
    
    # Load Data
    print("Loading Dolly-15k Test Set...")
    try:
        dataset = load_dataset("databricks/dolly-15k", cache_dir="/data/dolly15k")
    except:
        dataset = load_dataset("json", data_dir="/data/dolly15k")
        
    # Specific Split: 11k Train, 1k Val, 500 Test
    # We must replicate the shuffle seed=42 to get the same test set
    shuffled_dataset = dataset["train"].shuffle(seed=42)
    test_dataset = shuffled_dataset.select(range(12000, 12500))
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    # Create DataLoader for batching
    def collate_fn(batch):
        prompts = []
        metadata = []
        for example in batch:
            instruction = example['instruction']
            context = example['context']
            reference = example['response']
            
            if context:
                prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            prompts.append(prompt)
            metadata.append({
                "instruction": instruction,
                "context": context,
                "reference": reference
            })
        return prompts, metadata

    batch_size = 8
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )
    
    # Generation Loop
    results = []
    print(f"Generating responses for {len(test_dataset)} examples (Batch Size: {batch_size})...")
    
    for prompts, meta_batch in tqdm(test_loader):
        # Tokenize batch
        max_input_len = 1024 - 128
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_input_len
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode batch
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for gen_text, meta in zip(generated_texts, meta_batch):
            # Extract response part
            response_start = gen_text.find("### Response:")
            if response_start != -1:
                generated_response = gen_text[response_start + len("### Response:"):].strip()
            else:
                generated_response = gen_text
                
            results.append({
                "instruction": meta["instruction"],
                "context": meta["context"],
                "reference": meta["reference"],
                "generated": generated_response
            })
        
        # Save intermediate (every few batches)
        if len(results) % 100 < batch_size:
             with open("dolly_eval_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Final Save
    with open("dolly_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Generation Complete. Results saved to dolly_eval_results.json")
    
    # Calculate ROUGE Scores
    print("Calculating ROUGE Scores...")
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        
        predictions = [r["generated"] for r in results]
        references = [r["reference"] for r in results]
        
        scores = rouge.compute(predictions=predictions, references=references)
        
        print("\n=== Evaluation Results ===")
        print(f"ROUGE-1: {scores['rouge1']:.4f}")
        print(f"ROUGE-2: {scores['rouge2']:.4f}")
        print(f"ROUGE-L: {scores['rougeL']:.4f}")
        print(f"ROUGE-Lsum: {scores['rougeLsum']:.4f}")
        
        # Save scores
        with open("dolly_eval_scores.json", "w") as f:
            json.dump(scores, f, indent=2)
            
    except ImportError:
        print("WARNING: 'evaluate' or 'rouge_score' library not found. Skipping score calculation.")
        print("To enable scoring, run: pip install evaluate rouge_score")
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")

if __name__ == "__main__":
    main()
