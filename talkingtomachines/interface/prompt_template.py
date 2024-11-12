import pandas as pd
import argparse
import warnings
import ast
import re
from tqdm import tqdm
from talkingtomachines.interface.validate_template import *
from talkingtomachines.interface.initialize_experiment import initialize_experiment

PROMPT_TEMPLATE_SHEETS = [
    "experimental_setting",
    "treatments",
    "agent_roles",
    "prompts_template",
    "constants",
    "agent_profiles",
]


def extract_experimental_setting(template_file_path: str, sheet_name: str) -> dict:
    """Extracts the experimental setting from a specified worksheet in an Excel file.

    Args:
        template_file_path (str): The file path to the Excel template.
        sheet_name (str): The name of the worksheet containing the experimental settings.

    Returns:
        dict: A dictionary representation of the experimental settings, where the keys are the
              values from the first column and the values are the corresponding values from the
              second column.

    Raises:
        ValueError: If the mandatory fields are not present in the experimental setting worksheet.
    """
    # Read the experimental setting worksheet into a DataFrame
    experimental_setting_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the experimental setting worksheet
    validate_experimental_settings_sheet(experimental_setting_df)

    # Convert the experimental setting worksheet to a dictionary
    experimental_setting_dict = experimental_setting_df.set_index(
        experimental_setting_df.columns[0]
    ).to_dict()[experimental_setting_df.columns[1]]

    return experimental_setting_dict


def extract_treatments(template_file_path: str, sheet_name: str) -> dict:
    """Extracts treatment data from an Excel worksheet and returns it as a dictionary.

    Args:
        template_file_path (str): The file path to the Excel template.
        sheet_name (str): The name of the sheet within the Excel file to extract data from.

    Returns:
        dict: A dictionary containing the treatment data with the first column as keys and the second column as values.

    Raises:
        ValueError: If mandatory fields are not present in the treatments worksheet.
    """
    # Read the treatment worksheet into a DataFrame
    treatments_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the treatment worksheet
    validate_treatments_sheet(treatments_df)

    # Convert the treatment worksheet to a dictionary
    treatments_dict = treatments_df.set_index(treatments_df.columns[0]).to_dict()[
        treatments_df.columns[1]
    ]

    return {"treatments": treatments_dict}


def extract_agent_roles(template_file_path: str, sheet_name: str) -> dict:
    """Extracts agent roles from a specified Excel worksheet and returns them as a dictionary.

    Args:
        template_file_path (str): The file path to the Excel template.
        sheet_name (str): The name of the sheet within the Excel file that contains the agent roles.

    Returns:
        dict: A dictionary with a single key "agent_roles" mapping to another dictionary where the keys are the
              values from the first column of the worksheet and the values are the corresponding values from the
              second column.

    Raises:
        ValueError: If mandatory fields are not present in the agent roles worksheet.
    """
    # Read the agent role worksheet into a DataFrame
    agent_roles_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the treatment worksheet
    validate_agent_roles_sheet(agent_roles_df)

    # Convert the agent role worksheet to a dictionary
    agent_roles_dict = agent_roles_df.set_index(agent_roles_df.columns[0]).to_dict()[
        agent_roles_df.columns[1]
    ]

    return {"agent_roles": agent_roles_dict}


def generate_regex_for_response_options(response_options: str) -> list:
    """Generates a list of regex patterns based on the provided response options.

    This function takes a string of response options separated by semicolons and
    generates corresponding regex patterns. The response options can be either
    categorical values or numerical ranges. Numerical ranges are specified using
    a colon (e.g., "1:10" for values between 1 and 10).

    Args:
        response_options (str): A string containing response options separated by semicolons.
                                Numerical ranges can be specified using a colon.

    Returns:
        list: A list of regex patterns corresponding to the provided response options.

    Raises:
        ValueError: If a numerical range is specified incorrectly.
    """
    if pd.isnull(response_options):
        return response_options

    options = response_options.split(";")
    regex_patterns = []

    for option in options:
        option = option.strip()

        if ":" in option:
            start, end = option.split(":")
            start = start.strip()
            end = end.strip()

            try:
                if start == "":
                    start = float("-inf")
                else:
                    start = int(start)

                if end == "":
                    end = float("inf")
                else:
                    end = int(end)

                # Create the regex pattern based on the bounds
                if start == float("-inf") and end == float(
                    "inf"
                ):  # No upper or lower bound
                    condition = r"\d+"
                elif start == float("-inf"):  # No lower bound, only upper bound
                    condition = rf"\b([0-9]|[1-9][0-9]{{0,{len(str(end)) - 1}}})\b"  # Matches up to 'end'
                elif end == float("inf"):  # No upper bound, only lower bound
                    condition = (
                        rf"\b({start}[0-9]*)\b"  # Matches 'start' and numbers after it
                    )
                else:  # Both upper and lower bounds defined
                    if start == end:
                        condition = rf"\b{start}\b"  # Exact match if start == end
                    else:
                        # Regex pattern for inclusive range from 'start' to 'end'
                        condition = rf"\b({start}|{end}|\d{{1,{max(len(str(start)), len(str(end)))}}})\b"

                regex_patterns.append(condition)

            except ValueError:
                warnings.warn(
                    f"The following prompt {option} contains ':' indicating a range of numerical values; however, it is not a valid range. This response option will be treated as a categorical option."
                )
                regex_patterns.append(rf"\b{re.escape(option.lower())}\b")
        else:
            # regex_patterns.append(re.escape(option.lower()))
            regex_patterns.append(rf"\b{re.escape(option.lower())}\b")

    combined_regex = "|".join(regex_patterns)
    return combined_regex


def insert_response_options(row: pd.Series) -> str:
    """Inserts response options into a prompt string if a placeholder is present.

    This function checks if the placeholder "{response_options}" is present in the
    "prompt" field of the given pandas Series row. If the placeholder is found, it
    replaces it with a formatted string of response options separated by commas and
    an "or" before the last option. The response options are expected to be in the
    "response_options" field of the row, separated by semicolons.

    Args:
        row (pd.Series): A pandas Series containing at least "prompt" and
                         "response_options" fields.

    Returns:
        str: The prompt string with the response options inserted, if the placeholder
             was present. Otherwise, returns the original prompt string.
    """
    if "{response_options}" in row["experiment_prompt"]:
        options = row["response_options"].split(";")
        return row["experiment_prompt"].replace(
            "{response_options}",
            f'{", ".join([f"{repr(option.strip())}" for option in options[:-1]])} or {repr(options[-1].strip())}',
        )

    else:
        return row["experiment_prompt"]


def extract_prompts(template_file_path: str, sheet_name: str) -> dict:
    """Extracts prompts from an Excel worksheet and returns them as a dictionary.

    This function reads a specified worksheet from an Excel file, validates the presence
    of mandatory fields, processes the prompts, and returns them in a structured format.

    Args:
        template_file_path (str): The file path to the Excel template file.
        sheet_name (str): The name of the worksheet to read from the Excel file.

    Returns:
        dict: A dictionary containing a list of prompts. Each prompt is represented as a dictionary
              with relevant fields and their corresponding values.

    Raises:
        ValueError: If mandatory fields are not present in the prompts_template worksheet.
    """
    # Read the prompt template worksheet into a DataFrame
    prompts_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the prompt template worksheet
    validate_prompts_sheet(prompts_df)

    # Extract the final version of the prompts. First take from "text_adapted", if not take from "text", else ""
    prompts_df["experiment_prompt"] = prompts_df["text_adapted"]
    prompts_df["experiment_prompt"] = prompts_df["experiment_prompt"].fillna(
        prompts_df["text"]
    )
    prompts_df["experiment_prompt"] = prompts_df["experiment_prompt"].fillna("")

    # Insert response options into prompts
    prompts_df["experiment_prompt"] = prompts_df.apply(insert_response_options, axis=1)

    # Parse response options to regex format
    prompts_df["response_options"] = prompts_df["response_options"].apply(
        generate_regex_for_response_options
    )

    # Drop the irrelevant columns
    prompts_df.drop(columns=["is_adapted", "text", "text_adapted"], inplace=True)

    # Convert from dataframe to a list of dictionary
    prompt_list = []
    for _, row in prompts_df.iterrows():
        row_dict = {key: value for key, value in row.items() if not pd.isnull(value)}
        prompt_list.append(row_dict)

    return {"experiment_prompts": prompt_list}


def extract_constants(template_file_path: str, sheet_name: str) -> dict:
    """Extracts constants from a specified worksheet in an Excel file and returns them as a dictionary.

    Args:
        template_file_path (str): The file path to the Excel template.
        sheet_name (str): The name of the worksheet containing the constants.

    Returns:
        dict: A dictionary with a single key "constants" mapping to another dictionary where:
            - The keys are the values from the first column of the worksheet.
            - The values are lists derived from the second column of the worksheet. If the value is a string representation of a list, it is converted to an actual list.

    Raises:
        ValueError: If mandatory fields are not present in the constants worksheet.
    """
    # Read the constants worksheet into a DataFrame
    constants_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the constants worksheet
    validate_constants_sheet(constants_df)

    # Convert the agent role worksheet to a dictionary
    constants_dict = constants_df.set_index(constants_df.columns[0]).to_dict()[
        constants_df.columns[1]
    ]

    # Convert string representations of lists to actual lists
    for key, value in constants_dict.items():
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            constants_dict[key] = [str(item) for item in ast.literal_eval(value)]
        else:
            constants_dict[key] = [value]

    return {"constants": constants_dict}


def extract_agent_profiles(template_file_path: str, sheet_name: str) -> dict:
    """Extracts agent profiles from an Excel sheet.

    Args:
        template_file_path (str): The file path to the Excel template.
        sheet_name (str): The name of the sheet to read from the Excel file.

    Returns:
        dict: A dictionary containing:
            - "agent_profiles_mapping" (dict): A dictionary mapping of the first row of the sheet.
            - "agent_profiles" (pd.DataFrame): A DataFrame containing the remaining data with new column headers.

    Raises:
        ValueError: If mandatory fields are not present in the agent_profiles worksheet.
    """
    # Read the specified sheet into a DataFrame
    agent_profiles_df = pd.read_excel(template_file_path, sheet_name=sheet_name)

    # Extract the first row and convert them to a dictionary
    agent_profiles_mapping = agent_profiles_df.iloc[0].to_dict()

    # Use the first row as the new column headers for the remaining data
    agent_profiles = agent_profiles_df.iloc[1:].reset_index(drop=True)
    agent_profiles.columns = agent_profiles_df.iloc[0]

    return {
        "agent_profiles_mapping": agent_profiles_mapping,
        "agent_profiles": agent_profiles,
    }


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Read the prompt template provided by the user"
    )
    parser.add_argument(
        "template_file_path", type=str, help="Path to the prompt template"
    )
    parser.add_argument(
        "test_mode",
        type=str,
        help="Boolean indicating whether the experiment will be run in test mode",
    )
    args = parser.parse_args()

    # Validate the provided directory path to the prompt template
    validate_prompt_template_path(args.template_file_path)

    # Validate the test mode
    args.test_mode = validate_test_mode(args.test_mode)

    # Read the prompt template in Excel format
    prompt_template = pd.ExcelFile(args.template_file_path)

    # Validate the required sheets in the prompt template
    validate_prompt_template_sheets(prompt_template, PROMPT_TEMPLATE_SHEETS)

    # Dictionary to store data from each worksheet
    prompt_template_data = {}

    # Iterate through each worksheet
    for sheet_name in prompt_template.sheet_names:
        if sheet_name == "experimental_setting":
            prompt_template_data.update(
                extract_experimental_setting(args.template_file_path, sheet_name)
            )
        elif sheet_name == "treatments":
            prompt_template_data.update(
                extract_treatments(args.template_file_path, sheet_name)
            )
        elif sheet_name == "agent_roles":
            prompt_template_data.update(
                extract_agent_roles(args.template_file_path, sheet_name)
            )
        elif sheet_name == "prompts_template":
            prompt_template_data.update(
                extract_prompts(args.template_file_path, sheet_name)
            )
        elif sheet_name == "constants":
            prompt_template_data.update(
                extract_constants(args.template_file_path, sheet_name)
            )
        elif sheet_name == "agent_profiles":
            prompt_template_data.update(
                extract_agent_profiles(args.template_file_path, sheet_name)
            )
        else:
            warnings.warn(
                f"{sheet_name} will be ignored as it is not one of the required prompt template sheets: {', '.join(PROMPT_TEMPLATE_SHEETS)}"
            )

    # Initialize experiment based on prompt template
    experiment_list = initialize_experiment(prompt_template_data)

    # Ask for user confirmation to run experiment
    user_input = (
        input(
            "Verify the experiment settings provided above and reply 'y' to start running the experiment. Otherwise, respond with anything else to terminate the experiment: "
        )
        .strip()
        .lower()
    )
    if user_input == "y":
        print("Experiment has started.")
        for idx, experiment in tqdm(enumerate(experiment_list)):
            experiment.run_experiment(test_mode=args.test_mode, version=idx + 1)
        print("Experiment is completed successfully.")
    else:
        print("Experiment is terminated by the user.")


if __name__ == "__main__":
    main()
