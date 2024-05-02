import configparser
import argparse
import logging
import os
import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored  

logging.basicConfig(level=logging.INFO)
config = configparser.ConfigParser()
config.read('config.ini')
API_KEY = config['DEFAULT']['OPENAI_API_KEY']
GPT_MODEL = config['DEFAULT']['GPT_MODEL']

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(client, messages, tools=None, tool_choice=None, model=GPT_MODEL,
                            max_tokens=150, temperature=0.7, top_p=1.0, frequency_penalty=0.0, 
                            presence_penalty=0.0):
    '''
    This function sends a request to the OpenAI API to generate a chat completion response

    Parameters:
    client (openai.Client): OpenAI API client
    messages (list): List of messages in the conversation
    tools (list): List of tools to use in the completion
    tool_choice (str): Tool choice to use in the completion
    model (str): Model to use for the completion
    max_tokens (int): Maximum number of tokens to generate
    temperature (float): Controls randomness: lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.
    top_p (float): Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered
    frequency_penalty (float): Adjusts the frequency of words in the response
    presence_penalty (float): Adjusts the presence of words in the response
    '''
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    '''
    This function pretty prints the conversation messages from the chat completion response

    Parameters:
    messages (list): List of messages in the conversation
    '''
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


def read_question():
    '''
    This function reads a question from the user and returns the question
    '''
    question = input("Ask a question: ")
    return question

def read_criteria():
    '''
    This function reads the criteria from the user and returns the criteria
    '''
    criteria = input("Enter the criteria: ")
    return criteria

def read_answer():
    '''
    This function reads the answer from the user and returns the answer
    '''
    answer = input("Enter the answer: ")
    return answer

def evaluate_answer(client, question, criteria, answer, use_feedback=False):
    '''
    This function provides a grade and feedback for a student's answer to a question

    Parameters:
    client (openai.Client): OpenAI API client
    question (str): The question to evaluate
    criteria (str): The criteria to use for evaluation
    answer (str): The student's answer to evaluate
    '''

    # Feedback
    feedback_prompt = f"You are simulating a teacher's assessment. You are to provide feedback to student's answer to this question:\n\n{question}\n\nYou should also uses these additional criterias when providing feedback:\n\n{criteria}\n\n.The feedback should be provided in the form of a bulleted list.\n\n"
    feedback_message = f"The student's answer:\n{answer}\n"
    
    messages = [
        {"role": "system", "content": feedback_prompt},
        {"role": "user", "content": feedback_message}
    ]

    response = chat_completion_request(client, messages)
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    pretty_print_conversation(messages)

    # Grade
    grade_prompt = f"You are simulating a teacher's assessment. You will be given answer to this question:\n\n{question}\n\n"
    if use_feedback:
        grade_prompt += "The input may also include feedback from a teacher to the question, which you can also use when grading.\n\n"
    grade_prompt = f"You should also use these additional criterias when evaluation the answer:\n\n{criteria}\n\nThe possible grades are: excellent, good, poor."
    grade_message = f"The student's answer to be evaluated:\n{answer}"
    if use_feedback:
        grade_message += f"\n\nFeedback from a teacher:\n{response.choices[0].message.content}"

    messages = [

        {"role": "system", "content": grade_prompt},
        {"role": "user", "content": grade_message}

    ]

    response = chat_completion_request(client, messages)
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    pretty_print_conversation(messages)


def main():
    '''
    This function is the main function that reads args, reads input and evaluates the answer
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="file", help="Type of input: file or console. Default is file")
    parser.add_argument("-t", "--type", action='store_true', help='Optional argument to specify if feedback should be used in grading.')

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

    evaluate_answer(client, question, criteria, answer, args.type)

if __name__ == '__main__':
    main()