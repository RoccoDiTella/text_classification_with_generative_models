import pdb
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


def joint_logprob(input_text = 'Había un avestrúz', for_loop=True):

    input_largo = tokenizer(input_text, return_tensors='pt')['input_ids']
    output_logits = model(input_largo)['logits']

    if for_loop:

        output_logprobs = F.log_softmax(output_logits[0],dim=1)
        total_lp = 0
        for i in range(len(input_largo[0])-1):
            token_que_viene = input_largo[0][i+1]
            total_lp += output_logprobs[i][token_que_viene]



    else:

        log_probs = F.log_softmax(output_logits, dim=-1)      # (B, T, V)

        # ignore first token’s prediction, predict tokens 1..T-1
        log_probs_next = log_probs[:, :-1, :]                 # (B, T-1, V)
        targets        = input_largo[:, 1:]                   # (B, T-1)

        # pick the log-prob of the actual next token at each position
        token_log_probs = log_probs_next.gather(
            dim=2,
            index=targets.unsqueeze(-1)                       # (B, T-1, 1)
        ).squeeze(-1)                                         # (B, T-1)

        total_lp_batch = token_log_probs.sum(dim=1)           # (B,)
        total_lp = total_lp_batch[0]


    return total_lp



joint_logprob('Había un avestrúz')


