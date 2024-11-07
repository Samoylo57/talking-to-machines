import os
import pandas as pd
import warnings


def validate_prompt_template_path(file_path: str) -> None:
    """Validates the provided file path to ensure it points to an existing file.

    Args:
        file_path (str): The path to the file that needs to be validated.

    Raises:
        ValueError: If the file does not exist at the provided path.
    """
    # Validate the provided directory path
    if not os.path.isfile(file_path):
        raise ValueError(f"Prompt template cannot be found in this path: {file_path}")


def validate_prompt_template_sheets(
    excel_file: pd.ExcelFile, required_sheet_list: list
) -> None:
    """Validates that all required sheets are present in the given Excel file.

    Args:
        excel_file (pd.ExcelFile): The Excel file to validate.
        required_sheet_list (list of str): A list of sheet names that are required to be present in the Excel file.

    Raises:
        ValueError: If any of the required sheets are missing from the Excel file.
    """
    # Validate that all required sheets are present in the Excel file
    missing_sheets = [
        sheet for sheet in required_sheet_list if sheet not in excel_file.sheet_names
    ]
    if missing_sheets:
        raise ValueError(
            f"Missing required sheets in prompt template: {', '.join(missing_sheets)}"
        )


def validate_test_mode(test_mode: str) -> bool:
    """Validates the test_mode input and returns a boolean value.

    Args:
        test_mode (str): The test mode input, expected to be 'True', 'true', 'False', or 'false'.

    Returns:
        bool: True if test_mode is 'True' or 'true', False if test_mode is 'False' or 'false'.
              If the input is invalid, a warning is issued and the function returns True by default.

    Raises:
        UserWarning: If the input is not one of the valid options.
    """
    # Convert the input to lowercase for case-insensitive comparison
    test_mode = test_mode.lower()

    # Check if the input is one of the valid options
    if test_mode == "true":
        return True
    elif test_mode == "false":
        return False
    else:
        warnings.warn(
            f"Invalid test_mode: {test_mode}. Expected 'True', 'true', 'False', or 'false'. By default, test mode is set to 'True'."
        )
        return True


def validate_experimental_settings_sheet(experimental_settings: pd.DataFrame) -> None:
    """Validates the experimental settings sheet to ensure it has the correct structure and required settings.

    Args:
        experimental_settings (pd.DataFrame): A DataFrame containing the experimental settings.
                                            It should have columns "experimental_setting" and "value".

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
        AssertionError: If any of the required experimental settings are missing from the "experimental_setting" column.
    """
    # Validate the column headers
    expected_columns = ["experimental_setting", "value"]
    assert (
        list(experimental_settings.columns) == expected_columns
    ), f"Invalid columns in experimental_settings sheet. Expected {expected_columns}, got {list(experimental_settings.columns)}"

    # Validate the experimental settings field
    valid_settings = [
        "experiment_id",
        "model_info",
        "api_endpoint",
        "num_agents_per_session",
        "num_sessions",
        "max_conversation_length",
        "treatment_assignment_strategy",
        "agent_assignment_strategy",
        "treatment_column",
        "session_column",
    ]
    for setting in valid_settings:
        assert (
            setting in experimental_settings["experimental_setting"].tolist()
        ), f"{setting} not found in experimental_setting column."


def validate_treatments_sheet(treatments: pd.DataFrame) -> None:
    """Validates the structure of the treatments DataFrame.

    This function checks if the treatments DataFrame has the expected column headers.
    It raises an assertion error if the columns do not match the expected columns.

    Args:
        treatments (pd.DataFrame): The DataFrame containing treatment data to be validated.

    Raises:
        AssertionError: If the columns of the treatments DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = ["treatment_label", "treatment_description"]
    assert (
        list(treatments.columns) == expected_columns
    ), f"Invalid columns in treatments sheet. Expected {expected_columns}, got {list(treatments.columns)}"


def validate_agent_roles_sheet(agent_roles: pd.DataFrame) -> None:
    """Validates the structure of the agent roles DataFrame.

    This function checks if the provided DataFrame has the expected column headers.
    It raises an assertion error if the columns do not match the expected structure.

    Args:
        agent_roles (pd.DataFrame): The DataFrame containing agent roles to be validated.

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = ["role_label", "role_description"]
    assert (
        list(agent_roles.columns) == expected_columns
    ), f"Invalid columns in agent_roles sheet. Expected {expected_columns}, got {list(agent_roles.columns)}"


def validate_prompts_sheet(prompts: pd.DataFrame) -> None:
    """Validates the structure of a DataFrame containing prompt data.

    This function checks if the DataFrame has the expected column headers.
    If the columns do not match the expected headers, an assertion error is raised.

    Args:
        prompts (pd.DataFrame): The DataFrame containing the prompt data to be validated.

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = [
        "id",
        "type",
        "is_adapted",
        "text",
        "text_adapted",
        "var_name",
        "var_type",
        "response_options",
        "response_validation",
    ]
    assert (
        list(prompts.columns) == expected_columns
    ), f"Invalid columns in prompts_template sheet. Expected {expected_columns}, got {list(prompts.columns)}"


def validate_constants_sheet(constants: pd.DataFrame) -> None:
    """Validates the structure of a DataFrame containing constants.

    This function checks that the DataFrame has the expected column headers: "name" and "value".
    If the columns do not match the expected headers, an assertion error is raised.

    Args:
        constants (pd.DataFrame): The DataFrame to validate.

    Raises:
        AssertionError: If the DataFrame does not have the expected columns.
    """
    # Validate the column headers
    expected_columns = ["name", "value"]
    assert (
        list(constants.columns) == expected_columns
    ), f"Invalid columns in constants sheet. Expected {expected_columns}, got {list(constants.columns)}"
