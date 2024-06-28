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
    read_question,
    read_criteria,
    read_answer,
    get_grade_prompt_and_message,
    get_feedback_prompt_and_message,
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


async def evaluate_answer(
    client,
    question,
    criteria,
    answer,
    provide_feedback=True,
    use_feedback=False,
    czech=False,
    logprobs=False,
) -> None:
    """
    Provides a grade and feedback for a student's answer to a question

    Parameters:
    client (openai.Client): OpenAI API client
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answer (str): The student's answer to evaluate
    provide_feedback (bool): Whether to provide feedback
    use_feedback (bool): Whether to use feedback in the grading
    czech (bool): Whether the input is in Czech language
    logprobs (bool): Whether to print logprobs
    """

    # Feedback
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
            seed=15,
        )

        response = await chat_completion_request(feedback_params)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        pretty_print_conversation(messages)

    feedback = response.choices[0].message.content if provide_feedback else None

    # Grade
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

    pretty_print_conversation(messages)
    if logprobs:
        print_logprobs(response.choices[0].logprobs.content[0].top_logprobs)


async def main():
    """
    The main function that reads args, reads input and evaluates the answer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        action="store_true",
        help="Optional arguement to specify if the input should be from the console.",
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

    if not args.feedback and args.type:
        print(
            "Feedback is required when using the type argument. Please provide feedback."
        )
        return

    client = openai.AsyncClient(api_key=OPENAI_KEY)

    question, criteria, answer = "", "", ""

    if not args.input:
        input_file = "data/answers.json"
        with open(input_file, "r") as file:
            data = json.load(file)
            # TODO change to get all questions
            data = data["data"][2]
            question = data["question"]
            criteria = data["criteria"]
            answer = data["answer"]
    else:
        question = read_question()
        criteria = read_criteria()
        answer = read_answer()

    await evaluate_answer(
        client,
        question,
        criteria,
        answer,
        args.feedback,
        args.type,
        args.czech,
        args.logprobs,
    )


if __name__ == "__main__":
    asyncio.run(main())
