'''
This is the main file of the application. It defines the FastAPI application and
the endpoints (evaluating and generating) based. It's based on the various scripts
in this directory.
'''

import configparser
import logging
import openai
from typing import Dict
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils import (
    get_feedback_prompt_and_message,
    get_grade_prompt_and_message,
    chat_completion_request,
    RequestParams,
    EvaluationRequest,
    GenerationRequest,
    get_generation_prompt_and_message,
)

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

async def generate_questions(
    client,
    text,
    mode="mixed",
    czech=False,
) -> None:
    gen_prompt, gen_message = get_generation_prompt_and_message(text, mode, czech)

    messages = [
        {"role": "system", "content": gen_prompt},
        {"role": "user", "content": gen_message},
    ]

    gen_params = RequestParams(
        client=client,
        max_tokens=3000,
        messages=messages,
        temperature=0.5,
        top_p=1.0,
        seed=15,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    response = await chat_completion_request(gen_params)
    return {"question": response.choices[0].message.content}


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

@app.post("/generate")
@app.post("/generate/")
async def generate(request: GenerationRequest, api_key: str = Depends(get_api_key)):
    try:
        result = await generate_questions(client, request.text, request.mode, request.czech)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
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
