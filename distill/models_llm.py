import torch
from transformers import AutoModelForCausalLM

def get_llm_models(teacher_name="gpt2", student_name="gpt2", device="cuda", dropout=0.1):
    """
    Returns Teacher and Student models.
    """
    # Teacher
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Student
    from transformers import AutoConfig
    student_config = AutoConfig.from_pretrained(student_name)
    
    # Apply Dropout
    student_config.resid_pdrop = dropout
    student_config.embd_pdrop = dropout
    student_config.attn_pdrop = dropout
    
    student = AutoModelForCausalLM.from_pretrained(student_name, config=student_config)
    student.to(device)
    student.train()
    
    return teacher, student
