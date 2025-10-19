import os, json
import pandas as pd
import numpy as np
from typing import Any
from datetime import date, datetime


def _json_serializer(obj):
    """
    Serialize objects into JSON-compatible formats.

    This function is used to handle objects that are not serializable by the default
    JSON encoder. It provides custom serialization for specific types, including
    `datetime`, `date`, `numpy` arrays, and `numpy` generic types. For any other
    unknown objects, it falls back to converting them to their string representation.

    Args:
        obj: The object to be serialized.

    Returns:
        A JSON-serializable representation of the input object.

    Supported Types:
        - `datetime` and `date`: Converted to ISO 8601 string format.
        - `numpy.ndarray`: Converted to a Python list.
        - `numpy.generic`: Converted to its scalar value.
        - Any other object: Converted to its string representation.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

    return str(obj)


def save_experiment(
    experiment: dict[str, Any], save_results_as_csv: bool = False
) -> None:
    """Save an experiment to a local JSON file in the experiment_results folder at the root directory.

    Args:
        experiment (dict[str, Any]): The experiment to be saved.
        save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

    Returns:
        None
    """
    os.makedirs("experiment_results", exist_ok=True)
    json_file_path = f"experiment_results/{experiment['experiment_id']}.json"
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(
            experiment, file, default=_json_serializer, ensure_ascii=False, indent=2
        )

    if save_results_as_csv:
        save_experiment_as_csv(json_file_path)


def parse_json_field(json_field: str):
    """Parses a JSON string field, removing optional Markdown-style JSON code block delimiters
    (e.g., ```json ... ```) if present, and attempts to decode the JSON string into a Python object.

    Args:
        json_field (str): The JSON string to parse. It may optionally include Markdown-style
                          code block delimiters.

    Returns:
        dict or list or str: The parsed JSON object (e.g., a dictionary or list) if the input
                             is valid JSON. If the input is not valid JSON, the original string
                             is returned.
    """
    try:
        if json_field.startswith("```json"):
            json_field = json_field[len("```json") :].strip()
        if json_field.endswith("```"):
            json_field = json_field[:-3].strip()

        return json.loads(json_field)

    except json.JSONDecodeError:
        return json_field


def save_experiment_as_csv(file_name: str) -> None:
    """Reads a JSON file containing experiment data, processes the data to extract relevant information,
    and saves the result as a CSV file.

    Args:
        file_name (str): The path to the JSON file containing the experiment data.
    """
    with open(file_name, "r") as file:
        json_output = json.load(file)

    result_dict = {}
    for _, session_info in json_output["sessions"].items():

        for role, subject in session_info["subjects"].items():
            if subject["role"] == "Facilitator":
                continue

            elif subject["role"] == "Summarizer":
                subject_id = f"{role}_{session_info['session_id']}"

            else:
                subject_id = subject["profile_info"]["ID"]

            result_dict[subject_id] = {
                "experiment_id": subject["experiment_id"],
                "session_id": subject["session_id"],
                "model_info": subject["model_info"],
                "temperature": subject["temperature"],
                "role": role,
                "treatment_label": session_info["treatment_label"],
                "experiment_context": subject["experiment_context"],
                "system_message": subject["system_message"],
            }

        for message in session_info["message_history"]:
            role_label = list(message.keys())[0]
            if role_label in ["Facilitator", "system"]:
                continue

            if role_label == "Summarizer":
                subject_id = f"Summarizer_{session_info['session_id']}"
            else:
                subject_id = message["subject_id"]

            if message.get("task_id", "") == "" and message.get("var_name", "") == "":
                continue

            elif message.get("var_name", "") != "":
                parsed_field = parse_json_field(json_field=message[role_label])
                result_dict[subject_id][message["var_name"]] = parsed_field

                if isinstance(parsed_field, dict):
                    if parsed_field.get("response", "") != "":
                        result_dict[subject_id][f"{message['var_name']}.response"] = (
                            parsed_field.get("response")
                        )
                    if parsed_field.get("reasoning", "") != "":
                        result_dict[subject_id][f"{message['var_name']}.reasoning"] = (
                            parsed_field.get("reasoning")
                        )
                    if parsed_field.get("speculation_score", "") != "":
                        result_dict[subject_id][
                            f"{message['var_name']}.speculation_score"
                        ] = parsed_field.get("speculation_score")

            else:
                parsed_field = parse_json_field(json_field=message[role_label])
                result_dict[subject_id][message["task_id"]] = parsed_field

                if isinstance(parsed_field, dict):
                    if parsed_field.get("response", "") != "":
                        result_dict[subject_id][f"{message['task_id']}.response"] = (
                            parsed_field.get("response")
                        )
                    if parsed_field.get("reasoning", "") != "":
                        result_dict[subject_id][f"{message['task_id']}.reasoning"] = (
                            parsed_field.get("reasoning")
                        )
                    if parsed_field.get("speculation_score", "") != "":
                        result_dict[subject_id][
                            f"{message['task_id']}.speculation_score"
                        ] = parsed_field.get("speculation_score")

    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    result_df.reset_index(drop=False, inplace=True)
    result_df.rename(columns={"index": "ID"}, inplace=True)
    result_df.sort_values(by="session_id", ascending=True, inplace=True)
    result_df.to_csv(file_name[:-5] + ".csv", index=False)
