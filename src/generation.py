"""
This script generates question convering the input text.
"""

import configparser
import argparse
import logging
import openai
import json
import asyncio
from utils import (
    get_generation_prompt_and_message,
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


async def generate_questions(
    client,
    text,
    mode="mixed",
    czech=False,
) -> None:
    """
    Generates a question based on the input text

    Parameters:
    client (openai.AsyncClient): OpenAI API client
    mode (str): The mode for generating the question
    text (str): The text to generate a question for
    czech (bool): Whether the input is in Czech language
    """

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
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    pretty_print_conversation(messages)


async def main():
    """
    The main function that reads args and generates a question based on the input text
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--czech",
        action="store_true",
        help="Optional argument to specify if the input is in Czech language.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Optional argument to specify the file path to read the input text from.",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["yn", "alt", "tf", "wh", "whmc", "cloze", "clozemc", "mixed"],
        help="The mode for generating the question",
    )

    args = parser.parse_args()

    client = openai.AsyncClient(api_key=OPENAI_KEY)

    text = ""
    if not args.file:
        text = input("Enter the text: ")
    else:
        with open(args.file, "r") as file:
            text = file.read()

    await generate_questions(client, text, czech=args.czech, mode=args.mode)


if __name__ == "__main__":
    asyncio.run(main())
