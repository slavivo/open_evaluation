# Evaluation of open-ended questions

This project uses OpenAI's GPT model to evaluate open-ended questions. It provides a grade (excellent/výborné, good/dobré, poor/špatné) and feedback for a student's answer to a question.

## Prerequisites

- Python 3.x
- openai
- configparser
- tenacity
- termcolor

## Installing

1. Clone the repository
2. Install the required Python packages

```sh
pip install openai configparser tenacity termcolor
```

3. Create a config.ini file in the src directory with your OpenAI API key and GPT model. The .gitignore file is already set up to ignore this file.

```ini
[DEFAULT]
OPENAI_API_KEY = your_openai_api_key
GPT_MODEL = your_gpt_model
```

## Running the program 

You can run the program from the command line with the following command:

```sh
python src/evaluation.py
```

You can use -i or --input to enable input from console (for now default behaviour is to take the first item from data/answers.json); -f or --feedback to specify if feedback should be provided; -t or --type if feedback should be used in grading; -c or --czech if the input is in Czech language.

Example of running the program with feedback, using feedback in grading and in czech:

```sh
python src/evaluation.py -f -t -c
```

## Output examples

### Example 1:

**Question:**
Explain what a neuron is, detailing how they transmit information and what unique features they have.

**Criteria:**
Must include the terms “synapse” and “action potential.” Must mention the role of neurotransmitters.

**Feedback:**
- Your explanation of what a neuron is, is correct. Good job on mentioning that neurons transmit information to other nerve, muscle, or gland cells.
- However, your response is too brief and doesn't cover all the required aspects of the question. The explanation of how neurons transmit information is missing.
- You mentioned "synapse," which is good, but you didn't explain what it is or how it works in the context of neurons. Remember, synapses are junctures at which a neuron communicates with another cell.
- The term "action potential" is not included in your answer. This term is crucial in describing how neurons transmit information.

**Grade:**
poor

### Example 2:

**Question:**
What are the components and structure of a molecule of DNA?

**Criteria:**
Mention base pairs, sugar, and phosphate. Describe that DNA is a double helix. Note that base pairs pair up in a specific way using hydrogen bond (AT and GC).

**Feedback:**
- You have correctly identified that DNA is a complex molecule with a double helix structure. Well done!
- Good job in mentioning the base pairs (AT and GC), as well as the sugar and phosphate that form the backbone of the DNA molecule.
- However, there is an error in the pairs you've mentioned. Adenine (A) pairs with Thymine (T) and Guanine (G) pairs with Cytosine (C), not the other way around. This is a key component of DNA structure and it's important to remember this.
- You've correctly noted that the base pairs are held together by hydrogen bonds. This is a crucial aspect of DNA's structure and function, and it's good to see.

**Grade:**
poor

### Example 3:

**Question:**
How can large language models introduce biases into student evaluation?

**Criteria:**
None provided

**Feedback:**
- You've correctly pointed out that biases in large language models (LLMs) can come from their training data. If the data used to train these models has inherent biases, toxic content, or inaccuracies, these issues can be reflected in the model's output.
- You've also made a good point about the limitations of LLMs in evaluating information outside their training set. This could indeed lead to the penalization of students who provide answers based on more recent information or knowledge that the model isn't familiar with.
- Your understanding of LLMs as autoregressive language models is accurate. They generate predictions based on past inputs and do not have a reliable mechanism for fact-checking or discerning the truth value of the information they generate.

**Grade:**
excellent