import time, warnings, openai, re
from typing import List, Any, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


RETRY_DELAY = 300
MAX_RETRIES = 5
OPENAI_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "o1",
    "o1-pro",
    "o3-pro",
    "o3",
    "o4-mini",
]
NO_TEMPERATURE_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-codex",
    "gpt-5-pro",
    "o1",
    "o1-pro",
    "o3-pro",
    "o3",
    "o4-mini",
]


def query_llm(
    llm_client: openai.OpenAI,
    model_info: str,
    message_history: List[dict],
    temperature: float,
) -> str:
    """Queries a LLM for a response based on the latest message history.

    Args:
        llm_client (openai.OpenAI): The LLM client.
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.
        temperature (float): The model temperature setting for the LLM.

    Returns:
        str: Response from the LLM.
    """
    if model_info in OPENAI_MODELS:
        return query_open_ai(
            llm_client=llm_client,
            model_info=model_info,
            message_history=message_history,
            temperature=temperature,
        )
    elif model_info in ["hf-inference"]:
        return query_hugging_face(
            llm_client=llm_client,
            message_history=message_history,
            temperature=temperature,
        )
    else:
        warnings.warn(
            f"{model_info} is not 'hf-inference' and not one of the OpenAI instruct models ({OPENAI_MODELS}). Defaulting to query OpenAI endpoint."
        )
        return query_open_ai(
            llm_client=llm_client,
            model_info=model_info,
            message_history=message_history,
            temperature=temperature,
        )


def _normalize_dropbox_url(url: str) -> str:
    """
    Normalize a Dropbox URL to ensure it points to a direct downloadable file.

    This function modifies Dropbox URLs to use the "dl.dropboxusercontent.com"
    host and ensures the query parameters are adjusted for direct file access
    by setting "raw=1" and removing the "dl" parameter if present.

    Args:
        url (str): The original URL to be normalized.

    Returns:
        str: The normalized URL pointing to a direct downloadable file.

    Notes:
        - Only URLs with "dropbox.com" or "dropboxusercontent.com" in the host
          are modified. Other URLs are returned unchanged.
        - If the input URL lacks a scheme, "https" is assumed.
    """
    p = urlparse(url)

    # Only touch dropbox domains
    host = (p.netloc or "").lower()
    if "dropbox.com" not in host and "dropboxusercontent.com" not in host:
        return url

    # Force raw host
    new_host = "dl.dropboxusercontent.com"

    # Build new query: keep existing params except 'dl', ensure raw=1
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    # Remove Dropbox's dl param (dl=0/1), prefer raw=1 for direct file
    q.pop("dl", None)
    q["raw"] = "1"

    new_url = urlunparse(
        (
            p.scheme or "https",
            new_host,
            p.path,
            p.params,
            urlencode(q, doseq=True),
            p.fragment,
        )
    )
    return new_url


def extract_image_url(message_content: str) -> Optional[str]:
    """
    Extracts an image URL from the given message content.

    This function attempts to identify and return an image URL from the provided
    message content string. It supports both Base64-encoded image URLs and generic
    HTTP(S) image URLs. The function applies heuristics to determine whether a URL
    is likely to point to an image.

    Args:
        message_content (str): The content of the message to parse for image URLs.

    Returns:
        Optional[str]: The extracted image URL if found, otherwise None.

    The extraction process includes:
    1. Searching for Base64-encoded image URLs.
    2. Identifying generic HTTP(S) URLs and applying heuristics to determine if they
       are likely to point to images. This includes:
       - Checking for common image file extensions (e.g., .png, .jpg, .gif).
       - Looking for query parameters that indicate image formats.
       - Recognizing known image hosting domains.
       - Handling Dropbox URLs and normalizing them to raw image links if applicable.
    3. Falling back to URLs that appear to belong to image-related folders (e.g.,
       "/images/", "/media/").

    Note:
        - If multiple URLs are found, the function prioritizes those that are more
          likely to be images based on the heuristics.
        - Dropbox URLs are normalized to their raw image format if applicable.
    """
    if not message_content:
        return None

    # 1) Extract Base64 encoded image URL
    m = re.search(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", message_content, re.IGNORECASE
    )
    if m:
        return m.group(0)

    # 2) Extract generic http(s) image URLs
    candidates = []
    for match in re.finditer(r"https?://[^\s<>()\"']+", message_content, re.IGNORECASE):
        url = match.group(0).rstrip(".,);]'\"")
        candidates.append(url)

    if not candidates:
        return None

    # Heuristics for image-y links (w/ or w/o file extensions)
    IMAGE_HOST_HINT = re.compile(
        r"(googleusercontent\.com|ggpht\.com|twimg\.com|fbcdn\.net|cdninstagram\.com|"
        r"pinimg\.com|imgur\.com|media\.amazonaws\.com|cloudfront\.net|dropboxusercontent\.com|dropbox\.com)",
        re.IGNORECASE,
    )

    def looks_like_image(url: str) -> bool:
        if re.search(r"\.(png|jpe?g|gif|webp|svg)(?:[?#]|$)", url, re.IGNORECASE):
            return True
        if re.search(
            r"(?:[?&])(format|ext|fm|type|content[_-]?type)=((image/)?(png|jpe?g|gif|webp|svg))",
            url,
            re.IGNORECASE,
        ):
            return True
        # Dropbox raw flags
        if "dropbox.com" in url and re.search(
            r"[?&](raw=1|dl=1)\b", url, re.IGNORECASE
        ):
            return True
        if IMAGE_HOST_HINT.search(url):
            return True
        return False

    # Prefer likely image links
    for url in candidates:
        if looks_like_image(url):
            # If it's Dropbox, normalize to raw
            if "dropbox.com" in url or "dropboxusercontent.com" in url:
                return _normalize_dropbox_url(url)
            return url

    # Fallback: paths that look like images folder
    for url in candidates:
        if re.search(r"/(img|imgs|images|photos|media)/", url, re.IGNORECASE):
            if "dropbox.com" in url or "dropboxusercontent.com" in url:
                return _normalize_dropbox_url(url)
            return url

    return None


def extract_image_url_from_message(message: str) -> List[str]:
    """
    Formats a message history by extracting image URLs and structuring the content.

    Args:
        message (str): The message, potentially containing an image URL.

    Returns:
        List[str]: A list of formatted message components. If an image URL is found in the
        message content, the list will include both a text component and an image component.
        Otherwise, it will include only a text component.
    """
    image_url = extract_image_url(message)
    if image_url:
        formatted_message = [
            {"type": "input_text", "text": message},
            {"type": "input_image", "image_url": image_url},
        ]
    else:
        formatted_message = [{"type": "input_text", "text": message}]

    return formatted_message


def format_message_history_for_response_api(message_history: List[dict]) -> List[dict]:
    """
    Formats a message history for use with a response API.
    This function processes a list of message dictionaries and reformats them
    based on their roles. Messages with the role "system" or "assistant" are
    wrapped in a specific structure, while messages with the role "user" are
    processed to extract image URLs. An error is raised for any unknown roles.

    Args:
        message_history (List[dict]): A list of message dictionaries, where each
            dictionary contains the keys "role" (str) and "content" (str).
    Returns:
        List[dict]: A list of reformatted message dictionaries. Each dictionary
            contains the keys "role" (str) and "content" (list or dict), where
            the content structure depends on the role.
    Raises:
        ValueError: If a message contains an unknown role. Valid roles are
            "system", "user", and "assistant".
    """
    formatted_message_history = []
    for message in message_history:
        if message["role"] == "system":
            formatted_message_history.append(
                {
                    "role": message["role"],
                    "content": [{"type": "input_text", "text": message["content"]}],
                }
            )
        elif message["role"] == "assistant":
            formatted_message_history.append(
                {
                    "role": message["role"],
                    "content": [{"type": "output_text", "text": message["content"]}],
                }
            )
        elif message["role"] == "user":
            formatted_message_history.append(
                {
                    "role": message["role"],
                    "content": extract_image_url_from_message(message["content"]),
                }
            )
        else:
            raise ValueError(
                f"Unknown message role: {message['role']}. Should be one of these options: 'system', 'user', 'assistant'."
            )

    return formatted_message_history


def query_open_ai(
    llm_client: openai.OpenAI,
    model_info: str,
    message_history: List[dict],
    temperature: float,
) -> Any:
    """Query OpenAI API with the provided prompt.

    Args:
        llm_client (openai.OpenAI): The LLM client.
        model_info (str): Information about the model.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.
        temperature (float): The model temperature setting for the LLM.

    Returns:
        Any: Response from the LLM.
    """
    formatted_message_history = format_message_history_for_response_api(message_history)

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            if model_info in NO_TEMPERATURE_MODELS:
                response = llm_client.responses.create(
                    model=model_info,
                    input=formatted_message_history,
                    stream=False,
                )
            else:
                response = llm_client.responses.create(
                    model=model_info,
                    input=formatted_message_history,
                    temperature=temperature,
                    stream=False,
                )
            return response.output_text

        except Exception as e:
            # Log the exception
            print(
                f"Error during OpenAI API call: {e}. Retrying in {RETRY_DELAY // 60} mins..."
            )
            time.sleep(RETRY_DELAY)

    return ""


def query_hugging_face(
    llm_client: openai.OpenAI, message_history: List[dict], temperature: float
) -> Any:
    """Query Hugging Face's dedicated inference API end point with the provided prompt.

    Args:
        llm_client (openai.OpenAI): The LLM client from OpenAI class.
        message_history (List[dict]): Contains the history of message exchanged between user and assistant.
        temperature (float): The model temperature setting for the LLM.

    Returns:
        Any: Response from the LLM.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            response = llm_client.chat.completions.create(
                model="tgi",
                messages=message_history,
                temperature=temperature,
                stream=False,
            )
            formatted_response = response.choices[0].message.content.strip()

            return formatted_response

        except Exception as e:
            # Log the exception
            print(
                f"Error during Hugging Face API call: {e}. Retrying in {RETRY_DELAY // 60} mins..."
            )
            time.sleep(RETRY_DELAY)
