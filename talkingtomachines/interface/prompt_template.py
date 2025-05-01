import pandas as pd
import argparse
import warnings
import ast
import concurrent.futures
from tqdm import tqdm
from talkingtomachines.interface.validate_template import *
from talkingtomachines.interface.initialize_experiment import initialize_experiment
from talkingtomachines.management.experiment import AItoAIInterviewExperiment

PROMPT_TEMPLATE_SHEETS = [
    "experimental_setting",
    "treatments",
    "agent_roles",
    "prompts_template",
    "agent_profiles",
    "constants",
]
SPECIAL_ROLES = ["Facilitator", "Summariser"]


def extract_experimental_setting(file_path: str, sheet_name: str) -> dict:
    """Extracts the experimental setting from a specified worksheet in the prompt template.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the worksheet containing the experimental settings.

    Returns:
        dict: A dictionary representation of the experimental settings, where the keys are the
              values from the first column and the values are the corresponding values from the
              second column.

    Raises:
        ValueError: If the mandatory fields are not present in the experimental setting worksheet.
    """
    # Read the experimental setting worksheet into a DataFrame
    experimental_setting_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the experimental setting worksheet
    validate_experimental_settings_sheet(experimental_setting_df)

    # Convert the experimental setting worksheet to a dictionary
    experimental_setting_dict = experimental_setting_df.set_index(
        experimental_setting_df.columns[0]
    ).to_dict()[experimental_setting_df.columns[1]]

    return experimental_setting_dict


def extract_treatments(file_path: str, sheet_name: str) -> dict:
    """Extracts treatment information from the prompt template and returns it as a dictionary.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the sheet within the prompt template file to extract data from.

    Returns:
        dict: A dictionary containing the treatment data with the first column as keys and the second column as values.

    Raises:
        ValueError: If mandatory fields are not present in the treatments worksheet.
    """
    # Read the treatment worksheet into a DataFrame
    treatments_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the treatment worksheet
    validate_treatments_sheet(treatments_df)

    # Convert the treatment worksheet to a dictionary
    treatments_dict = treatments_df.set_index(treatments_df.columns[0]).to_dict()[
        treatments_df.columns[1]
    ]

    return {"treatments": treatments_dict}


def extract_agent_roles(file_path: str, sheet_name: str) -> dict:
    """Extracts agent roles from a specified worksheet in the prompt template and return them as a dictionary.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the sheet within the prompt template that contains the agent roles.

    Returns:
        dict: A dictionary with a single key "agent_roles" mapping to another dictionary where the keys are the
              values from the first column of the worksheet and the values are the corresponding values from the
              second column.

    Raises:
        ValueError: If mandatory fields are not present in the agent roles worksheet.
    """
    # Read the agent role worksheet into a DataFrame
    agent_roles_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the treatment worksheet
    validate_agent_roles_sheet(agent_roles_df)

    # Convert the agent role worksheet to a dictionary
    agent_roles_dict = agent_roles_df.set_index(agent_roles_df.columns[0]).to_dict()[
        agent_roles_df.columns[1]
    ]

    return {"agent_roles": agent_roles_dict}


def parse_llm_text_column(llm_text: str, agent_list: list, task_id: str) -> dict:
    """Parses the llm_text column.

    If llm_text is a plain string (i.e. not a dictionary literal), this function
    creates a dictionary where the keys are the agents in agent_list (excluding those in SPECIAL_ROLES)
    and the value is the plain string.

    Otherwise, if llm_text is a string representation of a dictionary, it is parsed and then
    validated such that the parsed dictionary contains all of the required agent keys.
    If any are missing, a ValueError is raised.

    Args:
        llm_text (str): The text to parse.
        agent_list (list): A list of agent names.
        task_id (str): The task ID associated with the llm_text.

    Returns:
        dict: A dictionary mapping agent names to the llm_text.
    """
    # Create list of user-defined agents, excluding the special ones
    user_defined_agents = [agent for agent in agent_list if agent not in SPECIAL_ROLES]
    stripped_text = llm_text.strip()

    # If the text starts with "{" and ends with "}", assume it's a dictionary literal.
    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        try:
            agent_prompt_dict = ast.literal_eval(llm_text)
        except (ValueError, SyntaxError) as e:
            warnings.warn(
                f"Error parsing llm_text as dictionary: {e}. The prompt will be treated as a plain string and assigned to all user-defined agents."
            )
            agent_prompt_dict = {agent: llm_text for agent in user_defined_agents}

        # Check if all allowed agents are keys in the parsed dictionary.
        missing_agents = [
            agent for agent in user_defined_agents if agent not in agent_prompt_dict
        ]
        if missing_agents:
            raise ValueError(
                f"The parsed string from llm_text column (task_id: {task_id}) is missing the following agents: {missing_agents}"
            )
        return agent_prompt_dict

    # Otherwise, it's a plain string: build a dictionary mapping each user-defined agent to that string.
    else:
        return {agent: llm_text for agent in user_defined_agents}


def extract_prompts(file_path: str, sheet_name: str, agent_list: list) -> dict:
    """Extracts prompts from a specified worksheet in the prompt template and returns them as a dictionary.

    This function reads a specified worksheet from the prompt template file, validates the presence
    of mandatory fields, processes the prompts, and returns them in a structured format.

    Args:
        file_path (str): The file path to the prompt template file.
        sheet_name (str): The name of the worksheet to read from the prompt template file.
        agent_list (list): A list of agent roles to be used in the experiment.

    Returns:
        dict: A dictionary containing a list of prompts. Each prompt is represented as a dictionary
              with relevant fields and their corresponding values.

    Raises:
        ValueError: If mandatory fields are not present in the prompts_template worksheet.
    """
    # Read the prompt template worksheet into a DataFrame
    prompts_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Validate the presence of mandatory fields in the prompt template worksheet
    validate_prompts_sheet(prompts_df)

    # Convert the prompts to a list of dictionaries
    prompts_list = prompts_df[
        [
            "task_id",
            "type",
            "task_order",
            "llm_text",
            "var_name",
            "var_type",
            "response_options",
            "randomize_response_order",
            "validate_response",
            "generate_speculation_score",
        ]
    ].to_dict(orient="records")

    # Parse llm_text column from string format to appropriate Python format
    for prompt_dict in prompts_list:
        prompt_dict["llm_text"] = parse_llm_text_column(
            llm_text=prompt_dict["llm_text"],
            agent_list=agent_list,
            task_id=prompt_dict["task_id"],
        )

    # Parse response_options column from string format to appropriate Python format
    for prompt_dict in prompts_list:
        if (
            prompt_dict["type"] == "context"
        ):  # No responses are expected for context prompts
            continue

        try:
            prompt_dict["response_options"] = ast.literal_eval(
                prompt_dict["response_options"]
            )
        except (ValueError, SyntaxError):
            # If the response options are not in a valid format, treat them as a string
            warnings.warn(
                f"The response_options field for task_id '{prompt_dict.get('task_id', 'Unknown ID')}' is not a valid Python literal expression and will be treated as a string: {prompt_dict['response_options']}"
            )
            continue

    return {"experiment_prompts": prompts_list}


def extract_constants(file_path: str, sheet_name: str) -> dict:
    """Extracts constants from a specified worksheet in the prompt template and returns them as a dictionary.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the worksheet containing the constants.

    Returns:
        dict: A dictionary with a single key "constants" mapping to another dictionary where:
            - The keys are the values from the first column of the worksheet.
            - The values are lists derived from the second column of the worksheet. If the value is a string representation of a list, it is converted to an actual list.

    Raises:
        ValueError: If mandatory fields are not present in the constants worksheet.
    """
    # Read the constants worksheet into a DataFrame
    constants_df = pd.read_excel(file_path, sheet_name=sheet_name)

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


def extract_agent_profiles(file_path: str, sheet_name: str) -> dict:
    """Extracts agent profiles from a specified worksheet from the prompt template.

    Args:
        file_path (str): The file path to the prompt template.
        sheet_name (str): The name of the sheet to read from the prompt template.

    Returns:
        dict: A dictionary containing:
            - "agent_profiles_mapping" (dict): A dictionary mapping of the first row of the sheet.
            - "agent_profiles" (pd.DataFrame): A DataFrame containing the remaining data with new column headers.

    Raises:
        ValueError: If mandatory fields are not present in the agent_profiles worksheet.
    """
    # Read the specified sheet into a DataFrame
    agent_profiles_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract the first row and convert them to a dictionary
    agent_profiles_mapping = agent_profiles_df.iloc[0].to_dict()

    # Use the first row as the new column headers for the remaining data
    agent_profiles = agent_profiles_df.iloc[1:].reset_index(drop=True)
    agent_profiles.columns = agent_profiles_df.iloc[0]

    return {
        "agent_profiles_mapping": agent_profiles_mapping,
        "agent_profiles": agent_profiles,
    }


def print_experimental_settings(experiment: AItoAIInterviewExperiment) -> None:
    """Prints the experimental settings for a given AI to AI interview experiment.

    Args:
        experiment (AItoAIInterviewExperiment): An instance of AItoAIInterviewExperiment containing all the settings and configurations for the experiment.
    """
    print(
        """
        Experiment Settings for {experiment_id}:
        {line_separator}
        Model Info: {model_info}
        API Endpoint (only valid when using HuggingFace Models): {api_endpoint}
        Agent Roles: {agent_roles}
        Number of Agents per Session (Including Interviewer): {num_agents_per_session}
        Number of Sessions: {num_sessions}
        Maximum Conversation Length: {max_conversation_length}
        Treatments: {treatments}
        Treatment Assignment Strategy: {treatment_assignment_strategy}
        Treatment Column (Only valid when using manual assignment strategy): {treatment_column}
        Session Assignment Strategy: {session_assignment_strategy}
        Session Column (Only valid when using manual assignment strategy): {session_column}
        Role Assignment Strategy: {role_assignment_strategy}
        Role Column (Only valid when using manual assignment strategy): {role_column}
        Random Seed: {random_seed}
        Experiment Prompts: {experiment_prompts}

        """.format(
            experiment_id=experiment.experiment_id,
            line_separator="=" * (25 + len(experiment.experiment_id)),
            model_info=experiment.model_info,
            api_endpoint=experiment.api_endpoint,
            agent_roles=experiment.agent_roles,
            num_agents_per_session=experiment.num_agents_per_session,
            num_sessions=experiment.num_sessions,
            max_conversation_length=experiment.max_conversation_length,
            treatments=experiment.treatments,
            treatment_assignment_strategy=experiment.treatment_assignment_strategy,
            treatment_column=experiment.treatment_column,
            session_assignment_strategy=experiment.session_assignment_strategy,
            session_column=experiment.session_column,
            role_assignment_strategy=experiment.role_assignment_strategy,
            role_column=experiment.role_column,
            random_seed=experiment.random_seed,
            experiment_prompts=experiment.experiment_prompts,
        )
    )


def run_experiment_wrapper(args: tuple) -> None:
    """Wrapper function to execute an experiment with the provided arguments.

    Args:
        args (tuple): A tuple containing the following elements:
            - experiment: An object with a `run_experiment` method to execute the experiment.
            - test_mode (bool): A flag indicating whether the experiment should run in test mode.
            - version (str): The version identifier for the experiment.

    Returns:
        None
    """
    experiment, test_mode, version = args
    experiment.run_experiment(
        test_mode=test_mode, version=version, save_results_as_csv=True
    )


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Read the prompt template provided by the user"
    )
    parser.add_argument(
        "prompt_template_file_path",
        type=str,
        default="",
        help="Path to the prompt template file",
    )
    parser.add_argument(
        "test_mode",
        type=str,
        default="",
        help="Boolean indicating whether the experiment will be run in test mode",
    )
    args = parser.parse_args()
    prompt_template_file_path = args.prompt_template_file_path
    test_mode = args.test_mode

    # Validate the provided directory path to the prompt template
    validate_prompt_template_path(file_path=prompt_template_file_path)

    # Validate the test mode
    test_mode = validate_test_mode(test_mode=test_mode)

    # Read the prompt template in Excel format
    prompt_template = pd.ExcelFile(prompt_template_file_path)

    # Validate the required sheets in the prompt template
    validate_prompt_template_sheets(
        excel_file=prompt_template, required_sheet_list=PROMPT_TEMPLATE_SHEETS
    )

    # Dictionary to store data from each worksheet
    prompt_template_dict = {}

    # Iterate through each worksheet
    for sheet_name in prompt_template.sheet_names:
        if sheet_name == "experimental_setting":
            prompt_template_dict.update(
                extract_experimental_setting(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "treatments":
            prompt_template_dict.update(
                extract_treatments(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "agent_roles":
            prompt_template_dict.update(
                extract_agent_roles(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "prompts_template":
            prompt_template_dict.update(
                extract_prompts(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "agent_profiles":
            prompt_template_dict.update(
                extract_agent_profiles(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        elif sheet_name == "constants":
            prompt_template_dict.update(
                extract_constants(
                    file_path=prompt_template_file_path, sheet_name=sheet_name
                )
            )
        else:
            warnings.warn(
                f"{sheet_name} will be ignored as it is not one of the required prompt template sheets: {', '.join(PROMPT_TEMPLATE_SHEETS)}"
            )

    # Initialize experiment based on prompt template
    experiment_list = initialize_experiment(prompt_template_dict=prompt_template_dict)

    # Print out experiment settings for user verification
    for experiment in experiment_list:
        print_experimental_settings(experiment)

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
        # Prepare a list of arguments for each experiment
        experiment_args = [
            (experiment, test_mode, idx + 1)
            for idx, experiment in enumerate(experiment_list)
        ]

        # Run experiments in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all experiments
            futures = [
                executor.submit(run_experiment_wrapper, arg) for arg in experiment_args
            ]

            # Update progress bar as each experiment completes
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass
        print("Experiment is completed successfully.")

    else:
        print("Experiment is terminated by the user.")


if __name__ == "__main__":
    main()
