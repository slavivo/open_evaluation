"""
This script evaluates a student's answer to a question based on the criteria provided.
"""

import configparser
import argparse
import logging
import openai
import json
import asyncio
from utils import (
    get_grade_prompt_and_message,
    get_feedback_prompt_and_message,
    get_summary_prompt_and_message,
    pretty_print_conversation,
    print_logprobs,
    chat_completion_request,
    RequestParams,
)

logging.basicConfig(level=logging.INFO)
config = configparser.ConfigParser()

# You need to create a config.ini file with your OpenAI API key and GPT model
config.read("src/config.ini")
OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]


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
    pretty_print_conversation(all_messages)

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

    pretty_print_conversation(all_messages)

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
    pretty_print_conversation(messages)

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

async def main():
    """
    The main function that reads args, reads input and evaluates the answer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Argument to specify the path to the .json file with questions and answers.",
    )
    parser.add_argument(
        "-f",
        "--feedback",
        action="store_true",
        help="Optional argument to specify if feedback should be provided.",
    )
    parser.add_argument(
        "-t",
        "--type",
        action="store_true",
        help="Optional argument to specify if feedback should be used in grading.",
    )
    parser.add_argument(
        "-c",
        "--czech",
        action="store_true",
        help="Optional argument to specify if the input is in Czech language.",
    )
    parser.add_argument(
        "-l",
        "--logprobs",
        action="store_true",
        help="Optional argument to specify if logprobs should be printed.",
    )

    args = parser.parse_args()

    if not args.path:
        print("Please specify the path to the .json file with questions and answers.")
        return

    client = openai.AsyncClient(api_key=OPENAI_KEY)

    question, criteria, answers = "", "", ""

    with open(args.path, "r") as file:
        data = json.load(file)
        question = data["question"]
        criteria = data["criteria"]
        answers = data["answers"]

    await evaluate_answer(
        client,
        question,
        criteria,
        answers,
        args.feedback,
        args.type,
        args.czech,
        args.logprobs,
    )


if __name__ == "__main__":
    asyncio.run(main())
