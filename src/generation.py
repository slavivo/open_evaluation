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

    parser.add_argument(
        "-t",
        "--type",
        choices=["keyword", "text"],
        help="The type of the input text",
    )

    args = parser.parse_args()

    if not args.type:
        print("Please specify the type of the input text.")
        return

    client = openai.AsyncClient(api_key=OPENAI_KEY)

    text = ""
    level = ""
    if not args.file or args.type == "keyword":
        if args.czech:
            text = input("Zadejte klíčová slova: ")
            level = input("Zadejte vzdělávací úroveň: ")
        else:
            text = input("Enter the keyword: ")
            level = input("Enter the educational level: ")
    else:
        with open(args.file, "r") as file:
            text = file.read()

    await generate_questions(client, text, level, czech=args.czech, mode=args.mode, keyword=args.type == "keyword") 


if __name__ == "__main__":
    asyncio.run(main())
