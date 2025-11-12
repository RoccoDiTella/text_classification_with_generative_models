import pdb
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


def joint_logprob(input_text = 'Había un avestrúz'):
    input_largo = tokenizer(input_text, return_tensors='pt')['input_ids']
    output_logits = model(input_largo)['logits']
    output_logprobs = F.log_softmax(output_logits[0],dim=1)


    total_lp = 0

    for i in range(len(input_largo[0])-1):
        token_que_viene = input_largo[0][i+1]
        total_lp += output_logprobs[i][token_que_viene]

    return total_lp



pdb.set_trace()


joint_logprob('Había un avestrúz')


