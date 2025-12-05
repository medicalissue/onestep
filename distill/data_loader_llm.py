import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def get_llm_loaders(dataset_name="wikitext", model_name="gpt2", batch_size=2, max_length=128, num_workers=4, data_root='./data'):
    """
    Returns LLM train and test loaders.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=data_root)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
        
        # Filter empty lines
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)
        
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            num_proc=num_workers, 
            remove_columns=["text"]
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        train_loader = DataLoader(
            tokenized_datasets["train"], 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            tokenized_datasets["test"], 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=data_collator,
            num_workers=num_workers
        )
        
        return train_loader, test_loader, tokenizer
    
    elif dataset_name == "dolly-15k":
        # User specified local path: /data/dolly15k
        # If it's a raw json/jsonl file, we might need "json" builder.
        # Assuming it's a HF dataset saved locally or a json file.
        # Let's try loading from local dir. If it fails, we might need "json" and data_files.
        # Given "databricks/dolly-15k", it's likely a JSONL.
        # Let's assume /data/dolly15k contains the dataset files.
        try:
            dataset = load_dataset("databricks/dolly-15k", cache_dir="/data/dolly15k")
        except:
             # Fallback if it's a local directory with json
            dataset = load_dataset("json", data_dir="/data/dolly15k")

        def tokenize_function(examples):
            # Format: "### Instruction: ... ### Context: ... ### Response: ..."
            texts = []
            for inst, ctx, resp in zip(examples['instruction'], examples['context'], examples['response']):
                if ctx:
                    text = f"### Instruction:\n{inst}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"
                else:
                    text = f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                texts.append(text + tokenizer.eos_token)
            
            return tokenizer(
                texts, 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
        
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            num_proc=num_workers, 
            remove_columns=["instruction", "context", "response", "category"]
        )
        
        # Specific Split: 11k Train, 1k Val, 500 Test
        # Total Dolly-15k is ~15k. We shuffle and take slices.
        shuffled_dataset = tokenized_datasets["train"].shuffle(seed=42)
        
        train_dataset = shuffled_dataset.select(range(0, 11000))
        val_dataset = shuffled_dataset.select(range(11000, 12000))
        test_dataset = shuffled_dataset.select(range(12000, 12500))
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=data_collator,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=data_collator,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=data_collator,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader, tokenizer

    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")
