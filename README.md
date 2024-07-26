# Project

This project uses OpenAI's GPT model for two tasks.

1. Evaluating open-ended questions. It provides grades (excellent/výborně, good/dobře, poor/špatně), feedbacks for the students' answers to a question and a summary of all the answers to show overall performance, trends etc..
2. Generating questions. Based on the input text it generates various types of questions that test the reader's complete understanding of the text.

## Prerequisites

- Python 3.x
- numpy
- openai
- configparser
- tenacity
- termcolor

## Installing

1. Clone the repository
2. Install the required Python packages

```sh
pip install openai configparser tenacity termcolor numpy
```

3. Create a config.ini file in the src directory with your OpenAI API key and GPT model. The .gitignore file is already set up to ignore this file.

```ini
[DEFAULT]
OPENAI_KEY = your_openai_api_key
GPT_MODEL = your_gpt_model
```

## Running evaluation

You can run the evaluation script from the command line with the following arguments.

1. -p or --path to specify the path to the .json file with questions and answers. Example of the file format can be found in the data directory.
2. -f or --feedback to specify if feedbacks should be provided
3. -t or --type if feedbacks should be used in grading
4. -c or --czech if the input is in Czech language
5. -l or --logprobs to show logprobs of the categories.

Example of running the program with feedback, using feedback in grading and in czech:

```sh
python src/evaluation.py -p data/answers.json -f -t -c
```

### Output example of one question

**Question:**
Explain what a neuron is, detailing how they transmit information and what unique features they have.

**Criteria:**
Must include the terms “synapse” and “action potential.” Must mention the role of neurotransmitters.

**Answer:**
Neurons are cells that transmit information to other nerve, muscle, or gland cells. They use synapses

**Feedback:**
- Your explanation of what a neuron is, is correct. Good job on mentioning that neurons transmit information to other nerve, muscle, or gland cells.
- However, your response is too brief and doesn't cover all the required aspects of the question. The explanation of how neurons transmit information is missing.
- You mentioned "synapse," which is good, but you didn't explain what it is or how it works in the context of neurons. Remember, synapses are junctures at which a neuron communicates with another cell.
- The term "action potential" is not included in your answer. This term is crucial in describing how neurons transmit information.

**Grade:**
poor

### Output example in Czech

**Otázka:**
Vysvětlete, co je to neuron, jak přenáší informace a jaké má jedinečné vlastnosti.

**Kritéria:**
Musí obsahovat pojmy ‚synapse‘ a ‚akční potenciál‘.“ Musí zmínit úlohu neurotransmiterů.

**Odpověď:**
Neurony jsou buňky, které předávají informace jiným nervovým, svalovým nebo žlázovým buňkám. Používají synapse.

**Zpětná vazba:**
- Vaše odpověď je příliš stručná a chybí v ní podrobnosti. Neuron je základní jednotkou nervového systému a přenáší informace prostřednictvím elektrických a chemických signálů. Synapse je místo, kde dochází k přenosu informací mezi neurony.
- Musíte zmínit akční potenciál, který je elektrickým impulzem, který cestuje po neuronu a přenáší informace.
- Nezapomněli jste zmínit úlohu neurotransmiterů.

**Hodnocení:**
špatně

## Running generation

You can run the generation script from the command line with the following command:

```sh
python src/generation.py -m mixed
```

First you need to specify the form of generation. The program can either generate questions based on a text file describing the problem (-t text) or based on a given keyword(s) (-t keyword). The first variant should containt questions that test all the important aspects of the text, while the second variant should generate questions that test the understanding of the given keyword(s).

You can also use -c or --czech to specify if the input text is in Czech language; -f or --file to specify path for file that contains input text and -m or --mode to specify the type of questions generated (yn - yes/no, alt - alternative, tf - true/false, wh - who/what/where/when/why/how, whmc - same as before but multi-choice, cloze - fill-in-the-blank, clozemc - same as before but multi-choice, mixed - all the above).

Example of running the program with input text in Czech from a file and generating yes/no questions:

```sh
python src/generation.py -c -t text -f data/input_cz.txt -m yn
```
and for generating questions based on given keyword(s):
```sh
python src/generation.py -c -t keyword -m alt
```

### Output example

**Input text:**

A neural network is a machine learning program, or model, that makes decisions in a manner similar to the human brain, by using processes that mimic the way biological neurons work together to identify phenomena, weigh options and arrive at conclusions.

Every neural network consists of layers of nodes, or artificial neurons—an input layer, one or more hidden layers, and an output layer. Each node connects to others, and has its own associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.
Neural networks rely on training data to learn and improve their accuracy over time. Once they are fine-tuned for accuracy, they are powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts. One of the best-known examples of a neural network is Google’s search algorithm.

Neural networks are sometimes called artificial neural networks (ANNs) or simulated neural networks (SNNs). They are a subset of machine learning, and at the heart of deep learning models.

**Generated wh questions:**
1. Who utilizes neural networks to improve accuracy and efficiency in tasks such as speech and image recognition?
2. What are the main components of a neural network?
3. When does a node in a neural network get activated?
4. Where can neural networks be applied to significantly reduce the time required for data classification and clustering?
5. Why are neural networks considered powerful tools in computer science and artificial intelligence?
6. How do neural networks mimic the decision-making process of the human brain?

### Output example in Czech

**Vstupní text:**
Neuronová síť je program nebo model strojového učení, který se rozhoduje podobně jako lidský mozek pomocí procesů, které napodobují způsob, jakým biologické neurony spolupracují při identifikaci jevů, zvažování možností a vyvozování závěrů.

Každá neuronová síť se skládá z vrstev uzlů neboli umělých neuronů - vstupní vrstvy, jedné nebo více skrytých vrstev a výstupní vrstvy. Každý uzel se připojuje k ostatním a má svou vlastní přiřazenou váhu a práh. Pokud je výstup některého jednotlivého uzlu vyšší než zadaná prahová hodnota, tento uzel se aktivuje a odešle data do další vrstvy sítě. V opačném případě nejsou další vrstvě sítě předána žádná data.
Neuronové sítě se učí na základě trénovacích dat a postupem času zlepšují svou přesnost. Jakmile jsou vyladěny na přesnost, jsou mocným nástrojem v informatice a umělé inteligenci, který nám umožňuje klasifikovat a shlukovat data vysokou rychlostí. Úlohy v oblasti rozpoznávání řeči nebo rozpoznávání obrazu mohou trvat minuty oproti hodinám ve srovnání s ruční identifikací prováděnou lidskými experty. Jedním z nejznámějších příkladů neuronové sítě je vyhledávací algoritmus společnosti Google.

Neuronové sítě se někdy nazývají umělé neuronové sítě (ANN) nebo simulované neuronové sítě (SNN). Jsou podmnožinou strojového učení a tvoří jádro modelů hlubokého učení.

**Generované otázky:**
1. Co je neuronová síť a jak napodobuje lidský mozek?
2. Z jakých vrstev se skládá každá neuronová síť?
3. Jak se rozhoduje, zda uzel v neuronové síti aktivuje a odešle data do další vrstvy?
4. Jakým způsobem se neuronové sítě učí a zlepšují svou přesnost?
5. Jaké jsou výhody použití neuronových sítí ve srovnání s ruční identifikací prováděnou lidskými experty?
6. Uveďte příklad aplikace neuronové sítě v reálném světě.
7. Pravda nebo nepravda: Neuronové sítě jsou synonymem pro strojové učení.
8. Jaký je rozdíl mezi neuronovou sítí a hlubokým učením?
9. Co znamenají zkratky ANN a SNN?
10. Kterou společností je známý vyhledávací algoritmus, který využívá neuronové sítě?
11. Kdy se uzel v neuronové síti neaktivuje?
12. Jaké typy úloh mohou neuronové sítě řešit?
13. Vyplňte mezeru: Neuronové sítě se učí na základě __________ dat.
14. Kde se neuronové sítě nacházejí v rámci strojového učení?
15. Proč jsou neuronové sítě považovány za mocný nástroj v informatice a umělé inteligenci?

For more examples, see the examples directory.
