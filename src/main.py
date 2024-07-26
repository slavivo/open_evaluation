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
    get_summary_prompt_and_message,
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
    level,
    mode="mixed",
    czech=False,
    keyword=False,
) -> None:
    """
    Generates a question based on the input text

    Parameters:
    client (openai.AsyncClient): OpenAI API client
    level (str): The educational level
    mode (str): The mode for generating the question
    text (str): The text to generate a question for
    czech (bool): Whether the input is in Czech language
    keyword (bool): Whether the input is a keyword
    """

    gen_prompt, gen_message = get_generation_prompt_and_message(text, level, mode, czech, keyword)

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


async def feedback_answer(question, criteria, answers, client, czech=False):
    all_messages = []
    feedbacks = []
    feedback_prompt, feedback_messages = get_feedback_prompt_and_message(
        question, criteria, answers, czech
    )
    for feedback_message in feedback_messages:
        messages = [
            {"role": "system", "content": feedback_prompt},
            {"role": "user", "content": feedback_message},
        ]
        feedback_params = RequestParams(
            client=client,
            max_tokens=500,
            messages=messages,
            temperature=0.5,
            top_p=0.5,
            seed=15,
        )
        response = await chat_completion_request(feedback_params)
        feedbacks.append(response.choices[0].message.content)
        all_messages.append(messages[1])
        all_messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

    return feedbacks

async def grade_answer(question, criteria, answers, client, feedbacks, use_feedback, czech, logprobs):
    grade_prompt, grade_messages = get_grade_prompt_and_message(
        answers,
        question,
        criteria,
        feedbacks,
        use_feedback,
        czech,
    )

    all_messages = []
    grades = []

    top_logprobs = 10 if logprobs else None

    for grade_message in grade_messages:
        messages = [
            {"role": "system", "content": grade_prompt},
            {"role": "user", "content": grade_message},
        ]

        grade_params = RequestParams(
            client=client,
            messages=messages,
            temperature=0.0,
            top_p=0.1,
            max_tokens=50,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            seed=15,
        )

        response = await chat_completion_request(grade_params)
        grades.append(response.choices[0].message.content)
        all_messages.append(messages[1])
        all_messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        if logprobs:
            all_messages[-1]["content"] += f"\n\nLogprobs:{response.choices[0].logprobs.content[0].top_logprobs}"

    return grades

async def summary_answer(question, criteria, answers, client, czech=False):
    summary_prompt, summary_message = get_summary_prompt_and_message(
        answers, question, criteria, czech
    )

    messages = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": summary_message},
    ]

    summary_params = RequestParams(
        client=client,
        max_tokens=500,
        messages=messages,
        temperature=0.5,
        top_p=0.5,
        seed=15,
    )

    response = await chat_completion_request(summary_params)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )

    return response.choices[0].message.content

async def evaluate_answer(
    client,
    question,
    criteria,
    answers,
    provide_feedback=True,
    use_feedback=False,
    czech=False,
    logprobs=False,
) -> None:
    """
    Provides a grade and feedback for a student's answer to a question

    Parameters:
    client (openai.AsyncClient): OpenAI API client
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answers (list): The students' answers to evaluate
    provide_feedback (bool): Whether to provide feedback
    use_feedback (bool): Whether to use feedback in the grading
    czech (bool): Whether the input is in Czech language
    logprobs (bool): Whether to print logprobs
    """

    # Feedback
    feedbacks = None
    if provide_feedback:
        feedbacks = await feedback_answer(question, criteria, answers, client, czech)

    # Grade
    grades = await grade_answer(question, criteria, answers, client, feedbacks, use_feedback, czech, logprobs)

    # Overall student summary
    summary = await summary_answer(question, criteria, answers, client, czech)

    return {
        "feedbacks": feedbacks,
        "grades": grades,
        "summary": summary,
    }

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return credentials.credentials

@app.post("/generate")
@app.post("/generate/")
async def generate(request: GenerationRequest, api_key: str = Depends(get_api_key)):
    try:
        result = await generate_questions(client, request.text, request.level, request.mode, request.czech, request.keyword)
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
            request.answers,
            request.provide_feedback,
            request.use_feedback,
            request.czech,
            request.logprobs,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
