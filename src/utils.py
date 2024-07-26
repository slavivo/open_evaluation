from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from typing import Tuple, Dict
import numpy as np
import openai
from pydantic import BaseModel
from typing import Optional

DEF_MODEL = "gpt-4o"

class EvaluationRequest(BaseModel):
    """
    This class defines the parameters for the evaluation request
    """
    question: str
    criteria: str
    answers: list[str]
    provide_feedback: Optional[bool] = True
    use_feedback: Optional[bool] = False
    czech: Optional[bool] = False
    logprobs: Optional[bool] = False

class GenerationRequest(BaseModel):
    """
    This class defines the parameters for the generation request
    """
    text: str
    level: str
    mode: Optional[str] = "mixed"
    czech: Optional[bool] = False
    keyword: Optional[bool] = False


class RequestParams:
    """
    This class defines the parameters for the request to the OpenAI API
    """

    def __init__(
        self,
        client,
        messages=None,
        tools=None,
        tool_choice=None,
        model=DEF_MODEL,
        max_tokens=300,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=None,
        logprobs=None,
        top_logprobs=None,
    ):
        self.client = client
        self.messages = messages
        self.tools = tools
        self.tool_choice = tool_choice
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    def get_params(self) -> Dict:
        """
        This function returns the parameters for the request to the OpenAI API
        """
        return {
            "messages": self.messages,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(
    params: RequestParams,
) -> openai.types.chat.chat_completion.ChatCompletion:
    """
    This function sends a request to the OpenAI API to generate a chat completion response

    Parameters:
    params (RequestParams): The parameters for the request to the OpenAI API
    """
    try:
        response = await params.client.chat.completions.create(**params.get_params())
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages) -> None:
    """
    This function pretty prints the conversation messages from the chat completion response

    Parameters:
    messages (list): List of messages in the conversation
    """
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "user":
            print(
                colored(f"user: {message['content']}\n", role_to_color[message["role"]])
            )
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['function_call']}\n",
                    role_to_color[message["role"]],
                )
            )
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "function":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )


def print_logprobs(logprobs):
    """
    This function prints the logprobs

    Parameters:
    logprobs (list): List of logprobs
    """

    categories_probs = []
    for logprob in logprobs:
        token = logprob.token.strip().lower()
        for i, (category, prob) in enumerate(categories_probs):
            if len(category) > len(token):
                if (category == token) or (
                    category.startswith(token) and len(token) >= 2
                ):
                    categories_probs[i] = (category, prob + np.exp(logprob.logprob))
                    break
            else:
                if (category == token) or (
                    token.startswith(category) and len(category) >= 2
                ):
                    categories_probs[i] = (token, prob + np.exp(logprob.logprob))
                    break
        else:
            categories_probs.append((token, np.exp(logprob.logprob)))

    for category, prob in categories_probs:
        print(f"Category: {category}, linear probability: {np.round(prob*100,2)}")


def get_grade_prompt_and_message(
    answers, question, criteria, feedbacks=None, use_feedback=False, czech=False
) -> Tuple[str, list[str]]:
    """
    Returns the prompt and message for grading the answer

    Parameters:
    answers (list): The students' answers to evaluate
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    feedbacks (list): The list of feedbacks to use for evaluation
    use_feedback (bool): Whether to use feedback in the evaluation
    czech (bool): Whether to use Czech language
    """
    grade_messages = []
    if not czech:
        grade_prompt = f"You are simulating a teacher's assessment. You will be given answer to this question:\n{question}\n"
        if use_feedback:
            grade_prompt += "The input may also include feedback from a teacher to the question, which you can also use when grading.\n"
        grade_prompt += f"You should also use these additional criterias when evaluation the answer:\n{criteria}\nThe possible grades are: excellent, good, poor. Your answer must be just the grade."
    else:
        grade_prompt = f"Simuluješ hodnocení učitele. Bude ti poskytnuta odpověď na tuto otázku:\n{question}\n"
        if use_feedback:
            grade_prompt += "Vstup může také zahrnovat zpětnou vazbu od učitele k otázce, kterou můžete také použít při hodnocení.\n"
        grade_prompt += f"Měl bys také použít tato dodatečná kritéria při hodnocení odpovědi:\n{criteria}\nMožné známky jsou: výborně, dobře, špatně. Tvoje odpověď musí být pouze známka."
    
    for answer, feedback in zip(answers, feedbacks):
        if not czech:
            grade_message = f"The student's answer to be evaluated:\n{answer}"
            if use_feedback:
                grade_message += f"\nFeedback from a teacher:\n{feedback}"
        else:
            grade_message = f"Odpověď studenta k hodnocení:\n{answer}"
            if use_feedback:
                grade_message += f"\nZpětná vazba od učitele:\n{feedback}"
        grade_messages.append(grade_message)

    return grade_prompt, grade_messages

def get_summary_prompt_and_message(
    answers, question, criteria, czech=False
) -> Tuple[str, str]:
    """
    Returns the prompt and message for summarizing the answers

    Parameters:
    answers (list): The students' answers to evaluate
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    czech (bool): Whether the input is in Czech language
    """
    if not czech:
        summary_prompt = f"Your task is to generate a consise summary of the students' answers to this question:\n{question}\nAnd these grading criterias:\n{criteria}\nThe summary should include:\n1. Overall Performance: Briefly describe how well the students performed according to the grading criteria.\n2. Common Struggles: Identify and summarize any common difficulties or misconceptions that students exhibited.\n3. Specific Areas of Difficulty: Highlight particular areas or aspects of the question where students struggled the most.\n4. General Observations: Note any patterns, trends, or noteworthy observations based on the students' answers."
        summary_message = "The students' answers:\n"
    else:
        summary_prompt = f"Tvým úkolem je vytvořit přehledné shrnutí odpovědí žáků na tuto otázku:\n{question}\nA tato kritéria hodnocení:\n{criteria}\nShrnutí by mělo obsahovat:\n1. Celkový výkon: Stručně popište, jak dobře si žáci vedli podle kritérií hodnocení.\n2. Společné problémy: Uveďte a shrňte všechny běžné obtíže nebo chybné představy, které žáci vykazovali.\n3. Specifické oblasti obtíží: Zdůrazněte konkrétní oblasti nebo aspekty otázky, v nichž měli studenti největší potíže.\n4. Obecné postřehy: Všimněte si všech vzorců, trendů nebo pozoruhodných postřehů vycházejících z odpovědí studentů."
        summary_message = "Odpovědi studentů:\n"
    for i in range(len(answers)):
            summary_message += f"{i+1}. {answers[i]}\n"

    return summary_prompt, summary_message


def get_feedback_prompt_and_message(
    question, criteria, answers, czech=False
) -> Tuple[str, list[str]]:
    """
    Returns the prompt and message for providing feedback

    Parameters:
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answers (list): The students' answers to evaluate
    czech (bool): Whether to use Czech language
    """
    if not czech:
        feedback_prompt = f"Your task is to give feedback to a student on this question:\n{question}\nYou are using Feedforward (formative assessment). Be brief and friendly, use emoticons. Encourage the student but avoid praise or flattery. When giving feedback, also use these additional criteria:\n{criteria}\nThe structure of your feedback must be as follows:\n- (In your answer it was great that: here you indicate what the pupil answered correctly, explain very briefly the evidence for the points above, either in bullet points or a short summary.)\n- (Based on your answer, I recommend that you focus this: here, give no more than two brief points that the pupil should avoid in the future, e.g. what was inaccurate, factually incorrect, etc., explain very briefly the evidence for the above points, either in bullet points or a short summary.)"
        feedback_messages = [f"The student's answer:\n{answer}\n" for answer in answers]
    else:
        feedback_prompt = f"Tvým úkolem je poskytnout žákovi zpětnou vazbu na tuto otázku:\n{question}\nPoužíváš metodu Feedforward (formativní hodnocení). Buď stručný a přátelský, používej emotikony. Povzbuzuj žáka, ale vyvaruj se chválení nebo pochlebování. Při poskytování zpětné vazby také použij tyto dodatečné kritéria:\n{criteria}\nStruktura tvé zpětné vazby musí být následující:\n- (Ve tvé odpovědi bylo skvělé, že: zde uveď, co žák/žákyně odpověděl/a správně, velmi stručně vysvětli důkazy k výše uvedeným bodům, buď formou odrážek, nebo krátkého shrnutí.)\n- (Na základě tvé odpovědi ti doporučuji zaměřit se na: zde uveď maximálně dva stručné body, kterých by se měl/a žák/žákyně v budoucnu vyvarovat, co např. bylo nepřesné, fakticky nesprávné apod., velmi stručně vysvětli důkazy k výše uvedeným bodům, buď formou odrážek, nebo krátkého shrnutí.)"
        feedback_messages = [f"Odpověď studenta:\n{answer}\n" for answer in answers]

    return feedback_prompt, feedback_messages

def get_generation_prompt_and_message(text, level, mode="mixed", czech=False, keyword=False) -> Tuple[str, str]:
    """
    Returns the prompt and message for generating questions based on the input text

    Parameters:
    text (str): The text to generate questions for
    level (str): The educational level to consider when generating the questions
    mode (str): The mode for generating the question
    czech (bool): Whether the input is in Czech language
    keyword (bool): Whether the input is a keyword
    """
    if not keyword:
        if mode == "yn":
            if not czech:
                gen_prompt = f"Your task is to make yes-no questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu ano-ne, které identifikují informace, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "alt":
            if not czech:
                gen_prompt = f"Your task is to make alternative questions (has two options to choose from) of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit alternativní otázky (na výběr jsou dvě možnosti) k identifikaci informací, které jsou explicitně uvedeny v daném textu. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "tf":
            if not czech:
                gen_prompt = f"Your task is to make true-false questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky pravda-nepravda k identifikaci informací, které jsou explicitně uvedeny v daném textu. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "wh":
            if not czech:
                gen_prompt = f"Your task is to make who, what, when, where, why, and how open-ended questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otevřené otázky typu kdo, co, kdy, kde, proč a jak, které identifikují informace, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "whmc":
            if not czech:
                gen_prompt = f"Your task is to make who, what, when, where, why, and how multiple choice questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu kdo, co, kdy, kde, proč a jak s více volbami, které identifikují informace, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "cloze":
            if not czech:
                gen_prompt = f"Your task is to make fill-in-the-blank open-ended questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otevřené otázky typu vyplňování mezery k identifikaci informací, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "clozemc":
            if not czech:
                gen_prompt = f"Your task is to make fill-in-the-blank multiple choice questions of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu vyplňování mezery s více volbami k identifikaci informací, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl."
        elif mode == "mixed":
            if not czech:
                gen_prompt = f"Your task is to make a mix of different types of open-ended or multiple-choice questions (yes-no, alternative, true-false, who, what, when, where, why, how, multiple choice, fill-in-the-blank) of identifying information that is explictly shown in the given text. These questions should test the reader's complete understanding of the text and shouldn't include the answer."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit směs různých typů otevřených otázek nebo otázek s více odpovědi (ano-ne, alternativní, pravda-nepravda, kdo, co, kdy, kde, proč, jak, s více volbami, vyplňování mezery) k identifikaci informací, jež jsou v daném textu explicitně uvedeny. Tyto otázky by měly prověřit, zda čtenář textu zcela porozuměl a neměli by obsahovat odpověď."
        gen_message = f"Text: {text}"
    else:
        if mode == "yn":
            if not czech:
                gen_prompt = f"Your task is to make yes-no questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu ano-ne z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "alt":
            if not czech:
                gen_prompt = f"Your task is to make alternative questions (has two options to choose from) from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit alternativní otázky (na výběr jsou dvě možnosti) z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "tf":
            if not czech:
                gen_prompt = f"Your task is to make true-false questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky pravda-nepravda z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "wh":
            if not czech:
                gen_prompt = f"Your task is to make who, what, when, where, why, and how open-ended questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otevřené otázky typu kdo, co, kdy, kde, proč a jak z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "whmc":
            if not czech:
                gen_prompt = f"Your task is to make who, what, when, where, why, and how multiple choice questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu kdo, co, kdy, kde, proč a jak s více volbami z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "cloze":
            if not czech:
                gen_prompt = f"Your task is to make fill-in-the-blank open-ended questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otevřené otázky typu vyplňování mezery z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "clozemc":
            if not czech:
                gen_prompt = f"Your task is to make fill-in-the-blank multiple choice questions from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit otázky typu vyplňování mezery s více volbami z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy."
        elif mode == "mixed":
            if not czech:
                gen_prompt = f"Your task is to make a mix of different types of open-ended or multiple-choice questions (yes-no, alternative, true-false, who, what, when, where, why, how, multiple choice, fill-in-the-blank) from the given keywords. These questions should test the reader's complete understanding of the topic related to the keywords and shouldn't include the answer."
            else:
                gen_prompt = f"Tvým úkolem je vytvořit směs různých typů otevřených otázek nebo otázek s více odpověďmi (ano-ne, alternativní, pravda-nepravda, kdo, co, kdy, kde, proč, jak, s více volbami, vyplňování mezery) z daných klíčových slov. Tyto otázky by měly prověřit, zda čtenář zcela rozumí tématu souvisejícímu s klíčovými slovy a neměly by obsahovat odpověď."
        if not czech:
            gen_prompt += "You will also be given the educational level to consider when generating the questions."
            gen_message = f"Keywords: {text}\nEducational level: {level}"
        else:
            gen_prompt += "Bude ti také poskytnut vzdělávací úroveň, který máš zohlednit při generování otázek."
            gen_message = f"Klíčová slova: {text}\nVzdělávací úroveň: {level}"


    return gen_prompt, gen_message
