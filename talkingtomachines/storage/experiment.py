import os
import pandas as pd
from typing import Any
import json


def save_experiment(
    experiment: dict[int, Any], save_results_as_csv: bool = False
) -> None:
    """Save an experiment to a local JSON file in the experiment_results folder at the root directory.

    Args:
        experiment (dict[int, Any]): The experiment to be saved.
        save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

    Returns:
        None
    """
    os.makedirs("experiment_results", exist_ok=True)
    json_file_path = f"experiment_results/{experiment['experiment_id']}.json"
    with open(json_file_path, "w") as file:
        json.dump(experiment, file)

    if save_experiment_as_csv:
        save_experiment_as_csv(json_file_path)


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

        for agent_role, agent in session_info["agents"].items():
            if agent["role"] == "Facilitator":
                continue

            elif agent["role"] == "Summarizer":
                agent_id = f"{agent_role}_{session_info['session_id']}"

            else:
                agent_id = agent["profile_info"]["ID"]

            result_dict[agent_id] = {
                "experiment_id": agent["experiment_id"],
                "session_id": agent["session_id"],
                "model_info": agent["model_info"],
                "role": agent_role,
                "treatment": agent["treatment"],
                "experiment_context": agent["experiment_context"],
                "system_message": agent["system_message"],
            }

        for message in session_info["message_history"]:
            role_label = list(message.keys())[0]
            if role_label in ["Facilitator", "system"]:
                continue

            if role_label == "Summarizer":
                agent_id = f"Summarizer_{session_info['session_id']}"
            else:
                agent_id = message["agent_id"]

            if message.get("task_id", "") == "" and message.get("var_name", "") == "":
                continue

            elif message.get("var_name", "") != "":
                result_dict[agent_id][message["var_name"]] = message[role_label]

            else:
                result_dict[agent_id][message["task_id"]] = message[role_label]

    result_df = pd.DataFrame.from_dict(result_dict, orient="index")
    result_df.reset_index(drop=False, inplace=True)
    result_df.rename(columns={"index": "ID"}, inplace=True)
    result_df.sort_values(by="session_id", ascending=True, inplace=True)
    result_df.to_csv(file_name[:-5] + ".csv", index=False)
