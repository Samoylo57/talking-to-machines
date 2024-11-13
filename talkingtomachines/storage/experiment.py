import os
import pandas as pd
from typing import Any
import json


def save_experiment(experiment: dict[int, Any], save_results_as_csv: bool = False) -> None:
    """Save an experiment to a local JSON file in the storage/experiment folder at the root directory.

    Args:
        experiment (dict[int, Any]): The experiment to be saved.
        save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

    Returns:
        None
    """
    os.makedirs("storage/experiment", exist_ok=True)
    json_file_path = f"storage/experiment/{experiment['experiment_id']}.json"
    with open(json_file_path, "w") as file:
        json.dump(experiment, file)

    if save_experiment_as_csv:
        save_experiment_as_csv(json_file_path)


def save_experiment_as_csv(file_name: str) -> None:
    """Reads a JSON file containing experiment data, processes the data to extract relevant information,
    and saves the result as an Excel file.
    
    Args:
        file_name (str): The path to the JSON file containing the experiment data.
    """
    with open(file_name, 'r') as file:
        json_output = json.load(file)

    role_labels = set([agent["role"] for agent in json_output["sessions"][next(iter(json_output["sessions"]))]["agents"]])

    result_dict = {}
    for _, session_info in json_output["sessions"].items():
        
        for profile in session_info["agent_profiles"]:
            result_dict[profile["ID"]] = {}

        for message in session_info["message_history"]:
            if message.get("agent_id", "") == "":
                continue
            else:
                role_label = role_labels.intersection(message.keys()).pop()
                if message.get("prompt_id","") == "" and message.get("var_name","") == "":
                    continue

                elif message.get("var_name","") != "":
                    result_dict[message["agent_id"]][message["var_name"]] = message[role_label]

                else:
                    result_dict[message["agent_id"]][message["prompt_id"]] = message[role_label]

    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    result_df.reset_index(drop=False, inplace=True)
    result_df.rename(columns={"index":"ID"}, inplace=True)
    result_df.sort_values(by="ID", ascending=True, inplace=True)
    result_df.to_excel(file_name[:-5] + ".xlsx", index=False)
