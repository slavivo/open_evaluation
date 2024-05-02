import configparser
import argparse
import logging
import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["DEFAULT"]["OPENAI_API_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]


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
        model=GPT_MODEL,
        max_tokens=150,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
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
        }


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    params: RequestParams,
) -> openai.types.chat.chat_completion.ChatCompletion:
    """
    This function sends a request to the OpenAI API to generate a chat completion response

    Parameters:
    params (RequestParams): The parameters for the request to the OpenAI API
    """
    try:
        response = params.client.chat.completions.create(**params.get_params())
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


def read_question() -> str:
    """
    This function reads a question from the user and returns the question
    """
    question = input("Ask a question: ")
    return question


def read_criteria() -> str:
    """
    This function reads the criteria from the user and returns the criteria
    """
    criteria = input("Enter the criteria: ")
    return criteria


def read_answer() -> str:
    """
    Reads the answer from the user and returns the answer
    """
    answer = input("Enter the answer: ")
    return answer


def get_grade_prompt_and_message(
    answer, question, criteria, feedback=None, use_feedback=False, czech=False
) -> Tuple[str, str]:
    """
    Returns the prompt and message for grading the answer

    Parameters:
    answer (str): The student's answer to evaluate
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    feedback (str): The feedback to use for evaluation
    use_feedback (bool): Whether to use feedback in the evaluation
    """
    if not czech:
        grade_prompt = f"You are simulating a teacher's assessment. You will be given answer to this question:\n\n{question}\n\n"
        if use_feedback:
            grade_prompt += "The input may also include feedback from a teacher to the question, which you can also use when grading.\n\n"
        grade_prompt = f"You should also use these additional criterias when evaluation the answer:\n\n{criteria}\n\nThe possible grades are: excellent, good, poor."

        grade_message = f"The student's answer to be evaluated:\n{answer}"
        if use_feedback:
            grade_message += f"\n\nFeedback from a teacher:\n{feedback}"
    else:
        grade_prompt = f"Simulujete hodnocení učitele. Budete poskytnuta odpověď na tuto otázku:\n\n{question}\n\n"
        if use_feedback:
            grade_prompt += "Vstup může také zahrnovat zpětnou vazbu od učitele k otázce, kterou můžete také použít při hodnocení.\n\n"
        grade_prompt = f"Měli byste také použít tato dodatečná kritéria při hodnocení odpovědi:\n\n{criteria}\n\nMožné známky jsou: výborně, dobře, špatně."

        grade_message = f"Odpověď studenta k hodnocení:\n{answer}"
        if use_feedback:
            grade_message += f"\n\nZpětná vazba od učitele:\n{feedback}"

    return grade_prompt, grade_message


def get_feedback_prompt_and_message(
    question, criteria, answer, czech=False
) -> Tuple[str, str]:
    """
    Returns the prompt and message for providing feedback

    Parameters:
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answer (str): The student's answer to evaluate
    use_feedback (bool): Whether to use feedback in the evaluation
    """
    if not czech:
        feedback_prompt = f"You are simulating a teacher's assessment. You are to provide feedback to student's answer to this question:\n\n{question}\n\nYou should also uses these additional criterias when providing feedback:\n\n{criteria}\n\n.The feedback should be provided in the form of a bulleted list.\n\n"
        feedback_message = f"The student's answer:\n{answer}\n"
    else:
        feedback_prompt = f"Simulujete hodnocení učitele. Máte poskytnout zpětnou vazbu na odpověď studenta na tuto otázku:\n\n{question}\n\nMěli byste také použít tyto dodatečné kritéria při poskytování zpětné vazby:\n\n{criteria}\n\n.Zpětná vazba by měla být poskytnuta ve formě odrážkového seznamu.\n\n"
        feedback_message = f"Odpověď studenta:\n{answer}\n"

    return feedback_prompt, feedback_message


def get_request_params(
    client,
    messages=None,
    tools=None,
    tool_choice=None,
    model=GPT_MODEL,
    max_tokens=150,
    temperature=0.7,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
) -> RequestParams:
    """
    This function returns the parameters for the request to the OpenAI API for providing feedback

    Parameters:
    client (openai.Client): OpenAI API client
    messages (list): List of messages in the conversation
    tools (list): List of tools to use in the request
    tool_choice (str): The tool choice to use in the request
    model (str): The model to use in the request
    max_tokens (int): The maximum number of tokens to generate
    temperature (float): The sampling temperature
    top_p (float): The nucleus sampling parameter
    frequency_penalty (float): The frequency penalty
    presence_penalty (float): The presence penalty
    """
    return RequestParams(
        client=client,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


def evaluate_answer(
    client, question, criteria, answer, use_feedback=False, czech=False
) -> None:
    """
    Provides a grade and feedback for a student's answer to a question

    Parameters:
    client (openai.Client): OpenAI API client
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answer (str): The student's answer to evaluate
    """

    # Feedback
    feedback_prompt, feedback_message = get_feedback_prompt_and_message(
        question, criteria, answer, czech
    )

    messages = [
        {"role": "system", "content": feedback_prompt},
        {"role": "user", "content": feedback_message},
    ]

    feedback_params = get_request_params(client, messages=messages)

    response = chat_completion_request(feedback_params)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    pretty_print_conversation(messages)

    # Grade
    grade_prompt, grade_message = get_grade_prompt_and_message(
        answer,
        question,
        criteria,
        response.choices[0].message.content,
        use_feedback,
        czech,
    )

    messages = [
        {"role": "system", "content": grade_prompt},
        {"role": "user", "content": grade_message},
    ]

    grade_params = get_request_params(client, messages=messages)

    response = chat_completion_request(grade_params)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    pretty_print_conversation(messages)


def main():
    """
    The main function that reads args, reads input and evaluates the answer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="file",
        help="Type of input: file or console. Default is file",
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

    args = parser.parse_args()

    client = openai.Client(api_key=API_KEY)

    question, criteria, answer = "", "", ""

    if args.input == "file":
        input_file = "../data/answers.json"
        with open(input_file, "r") as file:
            data = json.load(file)
            # TODO change to get all questions
            data = data["data"][0]
            question = data["question"]
            criteria = data["criteria"]
            answer = data["answer"]
    else:
        question = read_question()
        criteria = read_criteria()
        answer = read_answer()

    evaluate_answer(client, question, criteria, answer, args.type, args.czech)


if __name__ == "__main__":
    main()
