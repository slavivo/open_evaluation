# Evaluation of open-ended questions

This project uses OpenAI's GPT model to evaluate open-ended questions. It provides a grade (excellent/výborně, good/dobře, poor/špatně) and feedback for a student's answer to a question.

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

Note that the actual format of the output is different in the console.

### Example 1:

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

### Example 2:

**Question:**
What are the components and structure of a molecule of DNA?

**Criteria:**
Mention base pairs, sugar, and phosphate. Describe that DNA is a double helix. Note that base pairs pair up in a specific way using hydrogen bond (AT and GC).

**Answer:**
DNA is a complex molecule and it is shaped like a double helix ladder, where the rungs are base pairs ATGC and the scaffold is sugars and phosphates. The base pairs bind (A with G) and (C with T) using hydrogen bonds, which can be separated when the DNA is being read or duplicated.

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

**Answer:**
LLMs have bias because their training data can have toxic, biased, or inaccurate data in it. When evaluating students, LLMs could also penalize students that know information that is more recent or otherwise outside the LLM’s training set, which may appear to be inaccurate to the AI model. LLMs are also not designed to keep track of accurate information; they are autoregressive language models, and so they do not have a legitimate hold on fact and caution should be used when depending on an AI model for subtle communication.

**Feedback:**
- You've correctly pointed out that biases in large language models (LLMs) can come from their training data. If the data used to train these models has inherent biases, toxic content, or inaccuracies, these issues can be reflected in the model's output.
- You've also made a good point about the limitations of LLMs in evaluating information outside their training set. This could indeed lead to the penalization of students who provide answers based on more recent information or knowledge that the model isn't familiar with.
- Your understanding of LLMs as autoregressive language models is accurate. They generate predictions based on past inputs and do not have a reliable mechanism for fact-checking or discerning the truth value of the information they generate.

**Grade:**
excellent

## Output examples in Czech

### Příklad 1:

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

### Příklad 2:

**Otázka:**
Jaké jsou složky a struktura molekuly DNA?

**Kritéria:**
Uveďte páry bází, cukr a fosfát. Popište, že DNA je dvojitá šroubovice. Všimněte si, že páry bází se párují specifickým způsobem pomocí vodíkové vazby (AT a GC).

**Odpověď:**
DNA je složitá molekula a má tvar dvojitého šroubovicového žebříku, kde příčky tvoří páry bází ATGC a lešení tvoří cukry a fosfáty. Páry bází se spojují (A s G) a (C s T) pomocí vodíkových vazeb, které se mohou při čtení nebo duplikaci DNA oddělit.

**Zpětná vazba:**

Děkuji za vaši odpověď. Zde je několik bodů, které bych chtěl zdůraznit:

- Máte správně, že DNA má tvar dvojité šroubovice a je tvořena páry bází, cukrem a fosfátem. To je dobré.

- Při popisu párování bází však došlo k chybě. Adenin (A) se páruje s thymidinem (T) a guanin (G) se páruje s cytosinem (C), nikoli A s G a C s T. Je to klíčový aspekt struktury DNA, takže je důležité si to správně zapamatovat.

- Dobře jste si všiml, že páry bází se spojují pomocí vodíkových vazeb a mohou se oddělit během replikace DNA. To je správně a důležité pro pochopení, jak DNA funguje.

**Hodnocení:**
špatně

### Příklad 3:

**Otázka:**
Jak mohou velké jazykové modely vnášet do hodnocení studentů zkreslení?

**Kritéria:**
Nebylo zadáno

**Odpověď:**
Modely LLM mají zkreslení, protože jejich trénovací data mohou obsahovat toxická, zkreslená nebo nepřesná data. Při hodnocení studentů by LLM mohly penalizovat i studenty, kteří znají informace, které jsou novější nebo jinak mimo tréninkovou množinu LLM, což se modelu umělé inteligence může jevit jako nepřesné. LLM také nejsou určeny k tomu, aby sledovaly přesné informace; jsou to autoregresivní jazykové modely, a proto nemají legitimní držení faktů a při závislosti na modelu UI pro jemnou komunikaci je třeba postupovat obezřetně.

**Zpětná vazba:**

- Pozitivní: Váš výklad, jak mohou velké jazykové modely (LLM) vnášet zkreslení do hodnocení studentů, je jasný a přesný. Máte pravdu, že toxická, zkreslená nebo nepřesná data v trénovacích datech LLM mohou vést k nespravedlivému hodnocení.

- Pozitivní: Dobře jste identifikoval, jak může LLM penalizovat studenty, kteří znají informace mimo tréninkovou množinu LLM. 

- Pozitivní: Výborně jste si uvědomil, že LLM nejsou určeny k sledování přesných informací a nemají legitimní držení faktů. 

- K zlepšení: V budoucnu byste mohl poskytnout více konkrétních příkladů nebo scénářů, jak by mohlo dojít ke zkreslení hodnocení studentů. 

- K zlepšení: Můžete také diskutovat o možných řešeních nebo strategiích pro minimalizaci těchto zkreslení, aby byla vaše odpověď kompletnější.

**Hodnocení:**
dobře