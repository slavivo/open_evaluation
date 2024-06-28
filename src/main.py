'''
This is the main file of the application. It defines the FastAPI application and
the endpoints (evaluating and generating) based. It's based on the various scripts
in this directory.
'''

import configparser
import logging
import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Tuple, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# Read configuration
config = configparser.ConfigParser()
config.read("config.ini")
OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
API_KEY = config["DEFAULT"]["API_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]

# Initialize OpenAI client
client = openai.AsyncClient(api_key=OPENAI_KEY)

app = FastAPI()
security = HTTPBearer()

# Define request and response models for FastAPI
class EvaluationRequest(BaseModel):
    question: str
    criteria: str
    answer: str
    provide_feedback: Optional[bool] = True
    use_feedback: Optional[bool] = False
    czech: Optional[bool] = False
    logprobs: Optional[bool] = False

class RequestParams:
    def __init__(
        self,
        client,
        messages=None,
        tools=None,
        tool_choice=None,
        model=GPT_MODEL,
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
    try:
        response = await params.client.chat.completions.create(**params.get_params())
        return response
    except Exception as e:
        logging.error(f"Unable to generate ChatCompletion response: {e}")
        raise

def print_logprobs(logprobs):
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
    answer, question, criteria, feedback=None, use_feedback=False, czech=False
) -> Tuple[str, str]:
    if not czech:
        grade_prompt = f"You are simulating a teacher's assessment. You will be given answer to this question:\n\n{question}\n\n"
        if use_feedback:
            grade_prompt += "The input may also include feedback from a teacher to the question, which you can also use when grading.\n\n"
        grade_prompt += f"You should also use these additional criterias when evaluating the answer:\n\n{criteria}\n\nThe possible grades are: excellent, good, poor. Your answer must be just the grade."

        grade_message = f"The student's answer to be evaluated:\n{answer}"
        if use_feedback:
            grade_message += f"\n\nFeedback from a teacher:\n{feedback}"
    else:
        grade_prompt = f"Simulujete hodnocení učitele. Budete poskytnuta odpověď na tuto otázku:\n\n{question}\n\n"
        if use_feedback:
            grade_prompt += "Vstup může také zahrnovat zpětnou vazbu od učitele k otázce, kterou můžete také použít při hodnocení.\n\n"
        grade_prompt += f"Měli byste také použít tato dodatečná kritéria při hodnocení odpovědi:\n\n{criteria}\n\nMožné známky jsou: výborně, dobře, špatně. Tvoje odpověď musí být pouze známka."

        grade_message = f"Odpověď studenta k hodnocení:\n{answer}"
        if use_feedback:
            grade_message += f"\n\nZpětná vazba od učitele:\n{feedback}"

    return grade_prompt, grade_message

def get_feedback_prompt_and_message(
    question, criteria, answer, czech=False
) -> Tuple[str, str]:
    if not czech:
        feedback_prompt = f"Your task is to give feedback to a student on this question:\n\n{question}\n\nYou are using Feedforward (formative assessment). Be brief and friendly, use emoticons. Encourage the student but avoid praise or flattery. When giving feedback, also use these additional criteria:\n\n{criteria}\n\nThe structure of your feedback must be as follows:\n- (In your answer it was great that: here you indicate what the pupil answered correctly, explain very briefly the evidence for the points above, either in bullet points or a short summary.)\n- (Based on your answer, I recommend that you focus this: here, give no more than two brief points that the pupil should avoid in the future, e.g. what was inaccurate, factually incorrect, etc., explain very briefly the evidence for the above points, either in bullet points or a short summary.)"
        feedback_message = f"The student's answer:\n{answer}\n"
    else:
        feedback_prompt = f"Tvým úkolem je poskytnout žákovi zpětnou vazbu na tuto otázku:\n\n{question}\n\nPoužíváš metodu Feedforward (formativní hodnocení). Buď stručný a přátelský, používej emotikony. Povzbuzuj žáka, ale vyvaruj se chválení nebo pochlebování. Při poskytování zpětné vazby také použij tyto dodatečné kritéria:\n\n{criteria}\n\nStruktura tvé zpětné vazby musí být následující:\n- (Ve tvé odpovědi bylo skvělé, že: zde uveď, co žák/žákyně odpověděl/a správně, velmi stručně vysvětli důkazy k výše uvedeným bodům, buď formou odrážek, nebo krátkého shrnutí.)\n- (Na základě tvé odpovědi ti doporučuji zaměřit se na: zde uveď maximálně dva stručné body, kterých by se měl/a žák/žákyně v budoucnu vyvarovat, co např. bylo nepřesné, fakticky nesprávné apod., velmi stručně vysvětli důkazy k výše uvedeným bodům, buď formou odrážek, nebo krátkého shrnutí.)"
        feedback_message = f"Odpověď studenta:\n{answer}\n"

    return feedback_prompt, feedback_message

async def evaluate_answer(
    client,
    question,
    criteria,
    answer,
    provide_feedback=True,
    use_feedback=False,
    czech=False,
    logprobs=False,
) -> Dict:
    response_messages = []

    if provide_feedback:
        feedback_prompt, feedback_message = get_feedback_prompt_and_message(
            question, criteria, answer, czech
        )

        messages = [
            {"role": "system", "content": feedback_prompt},
            {"role": "user", "content": feedback_message},
        ]

        feedback_params = RequestParams(
            client=client,
            max_tokens=500,
            messages=messages,
            temperature=0.2,
            top_p=1.0,
            seed=15
        )

        response = await chat_completion_request(feedback_params)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        response_messages.append(messages)

    feedback = response.choices[0].message.content if provide_feedback else None

    grade_prompt, grade_message = get_grade_prompt_and_message(
        answer,
        question,
        criteria,
        feedback,
        use_feedback,
        czech,
    )

    messages = [
        {"role": "system", "content": grade_prompt},
        {"role": "user", "content": grade_message},
    ]

    top_logprobs = 10 if logprobs else None
    grade_params = RequestParams(
        client=client,
        messages=messages,
        temperature=0.0,
        max_tokens=50,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        seed=15,
    )

    response = await chat_completion_request(grade_params)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )

    response_messages.append(messages)

    return {
        "feedback": feedback if provide_feedback else None,
        "grade": response.choices[0].message.content,
        "logprobs": response.choices[0].logprobs.content[0].top_logprobs if logprobs else None,
        "messages": response_messages,
    }

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return credentials.credentials

@app.post("/evaluate/")
async def evaluate(request: EvaluationRequest, api_key: str = Depends(get_api_key)):
    try:
        result = await evaluate_answer(
            client,
            request.question,
            request.criteria,
            request.answer,
            request.provide_feedback,
            request.use_feedback,
            request.czech,
            request.logprobs,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
