import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"

# set up tokenizer and model
checkpoint= '/work/trainable-agents/models/Llama2_Voldemort'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

'''
Generate a response from Beethoven model
@param prompt: the prompt to generate response
@param character: the character to act like
@param loc_time: the location and time of the character (e.g. "Coffee Shop - Afternoon")
@param status: the status of the character
@return response: the generated response
'''
def generate_response(meta_prompt, character, loc_time, status):
    prompt = meta_prompt.format(character=character, loc_time=loc_time, status=status) + '\n\n'
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, do_sample=True, temperature=0.5, top_p=0.95, max_new_tokens=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# parameters
meta_prompt = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. 

The status of you is as follows:
Location: {loc_time}
Status: {status}

The interactions are as follows:"""
character = "Voldemort"
loc_time = "Library - Night"
status = f'{character} talks with Harry Potter about his favorite magic.'

# generate response
print(generate_response(meta_prompt, character, loc_time, status))