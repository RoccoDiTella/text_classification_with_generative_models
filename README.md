
# Generative Models are Classifiers Too

The goal of this project is to show that you can use a generative model as a classifier.

We'll fine tune an LLM once per category and evaluate the joint likelihood of a given text from each of these models. This will allow us to have a full posterior for classes and should minimize overfitting.



Plan:

1) Choose an LLM to finetune
  Llama 3.2 1B (NOT INSTRUCT)

2) Pipeline to evaluate log-probs of a given text

  TODO: There are two implementations now: ugly human for loop and sleek chatgpt vectors

3) Generate a synthetic dataset to fine-tune on

4) Finetune the LLM on a few classess

5) Use 3) to calculate P(text | class) for each class where each class is a model

6) Use 5) and Bayes to get the probability of the labels


Done till now:

Script that loads Llama and evaluates next token logits.

Next:

1) Shift logits to logprobs

2) Recover logprobs for all inputed tokens

----


