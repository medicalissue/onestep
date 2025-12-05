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
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")
