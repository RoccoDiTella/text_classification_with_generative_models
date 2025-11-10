
# Load model directly
import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# pdb.set_trace()

# input_hello = tokenizer('hello', return_tensors='pt')['input_ids']
# logits = model(input_hello)['logits']

input_largo = tokenizer('Había un aveztrúz', return_tensors='pt')['input_ids']

next_logits = model(input_largo)['logits'][0,-1]
next_token = tokenizer.decode(torch.argmax(next_logits))
print(next_token)



