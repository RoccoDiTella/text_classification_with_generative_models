
# Generative Models are Classifiers Too

The goal of this project is to show that you can use a generative model as a classifier.

We'll fine tune an LLM once per category and evaluate the joint likelihood of a given text from each of these models. This will allow us to have a full posterior for classes and should minimize overfitting.



1) Choose an LLM to finetune
  Llama 3.2 1B (NOT INSTRUCT)

2) Generate a synthetic dataset to fine-tune on

3) Pipeline to evaluate log-probs of a given text

4) Finetune the LLM on a few classess

5) Use 3) to calculate P(text | class) for each class where each class is a model

6) Use 5) and Bayes to get the probability of the labels
