import torch
from transformers import AutoModelForCausalLM

def get_llm_models(teacher_name="gpt2", student_name="gpt2", device="cuda"):
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
    student = AutoModelForCausalLM.from_pretrained(student_name)
    student.to(device)
    student.train()
    
    return teacher, student
