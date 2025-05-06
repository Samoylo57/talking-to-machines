import random
import pandas as pd
from typing import List, Any
from itertools import product


def simple_random_assignment_session(
    treatment_labels: List[str], session_id_list: List[Any], random_seed: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to each session using a simple random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        session_id_list (List[Any]): The list of session IDs for assignment.
        random_seed (int): The random seed for reproducibility.

    Returns:
        dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
    """
    # Set the seed for reproducibility
    random.seed(random_seed)

    treatment_assignment = {}
    for session_id in session_id_list:
        if not treatment_labels:
            treatment_assignment[session_id] = ""
        else:
            treatment_assignment[session_id] = random.choice(treatment_labels)

    return treatment_assignment


def complete_random_assignment_session(
    treatment_labels: List[Any], session_id_list: List[Any], random_seed: int
) -> dict[int, str]:
    """Assigns treatment labels randomly to a specified number of sessions using a complete random assignment strategy.

    Args:
        treatment_labels (List[str]): A list of treatment labels.
        session_id_list (List[Any]): The list of session IDs for assignment.
        random_seed (int): The random seed for reproducibility.

    Returns:
        dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
    """
    # Set the seed for reproducibility
    random.seed(random_seed)

    # Randomize the order of the session IDs
    randomised_sessions = session_id_list.copy()
    random.shuffle(randomised_sessions)

    num_treatments = len(treatment_labels)
    treatment_assignment = {}
    for i, session_id in enumerate(randomised_sessions):
        if not treatment_labels:
            treatment_assignment[session_id] = ""
        else:
            treatment_assignment[session_id] = treatment_labels[i % num_treatments]
    return treatment_assignment


# def full_factorial_assignment_session(
#     treatment_labels: List[List[str]], session_id_list: List[Any], random_seed: int
# ) -> dict[int, str]:
#     """Assigns treatment labels to sessions using a full factorial design assignment strategy.

#     Args:
#         treatment_labels (List[List[str]]): A list of lists containing the treatment labels.
#             Each inner list represents the possible labels for a specific treatment factor.
#         session_id_list (List[Any]): The list of session IDs for assignment.
#         random_seed (int): The random seed for reproducibility.

#     Returns:
#         dict[Any, str]: A dictionary where the keys represent the session id information and the values
#             represent the assigned treatment labels.
#     """
#     if not treatment_labels:
#         treatment_label_combinations = []
#     else:
#         treatment_label_combinations = list(product(*treatment_labels))

#     return complete_random_assignment_session(
#         treatment_labels=treatment_label_combinations, session_id_list=session_id_list, random_seed=random_seed
#     )


def manual_assignment_session(
    agent_profiles: pd.DataFrame,
    treatment_column: str,
    session_column: str,
    session_id_list: List[Any],
) -> dict[Any, str]:
    """Extract the session treatment dictionary pairs provided by the user.

    Args:
        agent_profiles (pd.DataFrame): A list of treatment labels.
        treatment_column (str): The column containing the assigned treatments.
        session_column (str): The column containing the session information.
        session_id_list (List[Any]): The list of session IDs for assignment.

    Returns:
        dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
    """
    session_treatment_dict = {}
    for session_id in session_id_list:
        session_treatment_set = set(
            agent_profiles[agent_profiles[session_column] == session_id][
                treatment_column
            ].tolist()
        )

        if len(session_treatment_set) == 1:
            session_treatment_dict[session_id] = session_treatment_set.pop()
        else:
            raise ValueError(
                f"Session {session_id} is assigned different treatments: {session_treatment_set}"
            )

    return session_treatment_dict
