import os
import pandas as pd


def validate_prompt_template_path(file_path: str) -> None:
    """Validates the provided file path to ensure it points to an existing Excel prompt template file.

    Args:
        file_path (str): The path to the prompt template file that needs to be validated.

    Raises:
        ValueError: If the file does not exist at the provided path or is not an Excel file.
    """
    # Validate the provided directory path
    if not os.path.isfile(file_path):
        raise ValueError(
            f"The prompt template cannot be found in the path you provided: {file_path}"
        )

    # Check if the file has a valid Excel extension
    valid_extensions = [".xlsx", ".xls"]
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(
            f"The file provided is not a valid Excel file. Expected extensions: {', '.join(valid_extensions)}"
        )


def validate_prompt_template_sheets(
    excel_file: pd.ExcelFile, required_sheet_list: list
) -> None:
    """Validates that all required sheets are present in the given prompt template file.

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
            f"The following sheets are missing from the prompt template: {', '.join(missing_sheets)}"
        )


def validate_experimental_settings_sheet(experimental_settings: pd.DataFrame) -> None:
    """Validates the experimental settings worksheet to ensure it has the correct structure and required settings.

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
        "temperature",
        "num_agents_per_session",
        "num_sessions",
        "max_conversation_length",
        "treatment_assignment_strategy",
        "treatment_column",
        "session_assignment_strategy",
        "session_column",
        "role_assignment_strategy",
        "role_column",
        "random_seed",
    ]
    for setting in valid_settings:
        assert (
            setting in experimental_settings["experimental_setting"].tolist()
        ), f"{setting} not found in experimental_setting worksheet."


def validate_treatments_sheet(treatments: pd.DataFrame) -> None:
    """Validates the structure of the treatments worksheet.

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
    """Validates the structure of the agent_roles worksheet.

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
    """Validates the structure of a prompts_template worksheet containing prompt data.

    This function checks if the DataFrame has the expected column headers.
    If the columns do not match the expected headers, an assertion error is raised.

    Args:
        prompts (pd.DataFrame): The DataFrame containing the prompt data to be validated.

    Raises:
        AssertionError: If the columns of the DataFrame do not match the expected columns.
    """
    # Validate the column headers
    expected_columns = [
        "task_id",
        "type",
        "task_order",
        "is_adapted",
        "human_text",
        "llm_text",
        "var_name",
        "var_type",
        "response_options",
        "randomize_response_order",
        "validate_response",
        "generate_speculation_score",
        "format_response",
    ]
    assert (
        list(prompts.columns) == expected_columns
    ), f"Invalid columns in prompts_template sheet. Expected {expected_columns}, got {list(prompts.columns)}"


def validate_constants_sheet(constants: pd.DataFrame) -> None:
    """Validates the structure of the constants worksheet.

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
