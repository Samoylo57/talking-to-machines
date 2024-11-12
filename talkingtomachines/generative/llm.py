from typing import List, Any
from openai import OpenAI


def query_llm(llm_client: Any, model_info: str, message_history: List[dict]) -> str:
    """Queries a LLM for a response based on the latest message history.

    Args:
        llm_client (Any): The LLM client.
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.

    Returns:
        str: Response from the LLM.
    """
    if model_info in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
        return query_open_ai(
            llm_client=llm_client,
            model_info=model_info,
            message_history=message_history,
        )
    elif model_info in ["hf-inference"]:
        return query_hugging_face(
            llm_client=llm_client,
            message_history=message_history,
        )
    else:
        # Log the exception
        print(f"Model type {model_info} is not supported.")
        return ""


def query_open_ai(
    llm_client: OpenAI, model_info: str, message_history: List[dict]
) -> str:
    """Query OpenAI API with the provided prompt.

    Args:
        llm_client (OpenAI): The LLM client from OpenAI class.
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.

    Returns:
        str: Response from the LLM.
    """
    try:
        response = llm_client.chat.completions.create(
            model=model_info, messages=message_history
        )
        return response.choices[0].message.content

    except Exception as e:
        # Log the exception
        print(f"Error during OpenAI API call: {e}")
        return ""


def query_hugging_face(llm_client: OpenAI, message_history: List[dict]) -> str:
    """Query Hugging Face's dedicated inference API end point with the provided prompt.

    Args:
        llm_client (OpenAI): The LLM client from OpenAI class.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.
        api_endpoint (str, optional): API endpoint to the LLM that is hosted externally.

    Returns:
        str: Response from the LLM.
    """
    try:
        response = llm_client.chat.completions.create(
            model="tgi", messages=message_history, stream=False
        )

        return response.choices[0].message.content

    except Exception as e:
        # Log the exception
        print(f"Error during Hugging Face API call: {e}")
        return ""
