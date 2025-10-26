import datetime, random, warnings, concurrent.futures
import pandas as pd
from collections import defaultdict
from typing import Any, List
from tqdm import tqdm
from talkingtomachines.generative.synthetic_subject import (
    ConversationalSyntheticSubject,
    ProfileInfo,
)
from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    manual_assignment_session,
)
from talkingtomachines.generative.prompt import (
    generate_session_system_message,
)
from talkingtomachines.storage.experiment import save_experiment

SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES = [
    "simple_random",
    "complete_random",
    "manual",
]
SUPPORTED_SESSION_ASSIGNMENT_STRATEGIES = [
    "random",
    "manual",
]
SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES = [
    "random",
    "manual",
]
SPECIAL_ROLES = ["Facilitator", "Summarizer"]
SUPPORTED_PROMPT_TYPES = [
    "context",
    "public_question",
    "private_question",
    "discussion",
]


class Experiment:
    """A class for constructing the base experiment class.

    Args:
        experiment_id (str): The unique ID of the experiment.

    Attributes:
        experiment_id (str): The unique ID of the experiment.
    """

    def __init__(self, experiment_id: str = ""):
        if experiment_id == "":
            self.experiment_id = self._generate_experiment_id()
        else:
            self.experiment_id = experiment_id

    def _generate_experiment_id(self) -> str:
        """Generates a unique ID for the experiment by concatenating the date and time information.

        Returns:
            str: Unique ID for the experiment as a base64 encoded string.
        """
        current_datetime = datetime.datetime.now()
        experiment_id = current_datetime.strftime("%Y%m%d_%H%M%S")

        return experiment_id


class AIConversationalExperiment(Experiment):
    """A class representing an AI conversational experiment. Inherits from the Experiment base class.

    This class extends the base `Experiment` class and provides additional functionality
    specific to AI conversational experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        include_backstories (bool, optional): Whether to include backstories in the subject profiles. Defaults to False.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided demographic_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
        include_backstories (bool, optional): Whether to include backstories in the subjects profiles.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        demographic_profiles: pd.DataFrame,
        experiment_context: str = "",
        experiment_id: str = "",
        hf_inference_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        include_backstories: bool = False,
    ):
        super().__init__(
            experiment_id,
        )

        self.model_info = self._check_model_info(model_info=model_info)
        self.temperature = self._check_temperature(temperature=temperature)
        self.experiment_context = experiment_context
        self.demographic_profiles = self._check_demographic_profiles(
            demographic_profiles=demographic_profiles
        )
        self.hf_inference_endpoint = hf_inference_endpoint
        self.max_conversation_length = self._check_max_conversation_length(
            max_conversation_length=max_conversation_length
        )
        self.treatments = self._check_treatments(treatments=treatments)
        self.treatment_assignment_strategy = self._check_treatment_assignment_strategy(
            treatment_assignment_strategy=treatment_assignment_strategy,
            treatment_column=treatment_column,
            session_assignment_strategy=session_assignment_strategy,
        )
        self.treatment_column = treatment_column
        self.session_assignment_strategy = self._check_session_assignment_strategy(
            session_assignment_strategy=session_assignment_strategy,
            session_column=session_column,
        )
        self.session_column = session_column
        self.role_assignment_strategy = self._check_role_assignment_strategy(
            role_assignment_strategy=role_assignment_strategy, role_column=role_column
        )
        self.role_column = role_column
        self.random_seed = random_seed
        self.include_backstories = include_backstories

    def _check_model_info(self, model_info: str) -> str:
        """Checks if the provided model_info is supported.

        Args:
            model_info (str): The model_info to be checked.

        Returns:
            str: The validated model_info.

        Raises:
            ValueError: If the provided model_info is empty.
        """
        if not isinstance(model_info, str) or not model_info.strip():
            raise ValueError("The model_info field must be a non-empty string.")

        return model_info.strip()

    def check_model_info(self, model_info: str) -> str:
        """Public wrapper for validating model identifiers.

        This helper mirrors the internal `_check_model_info` validation so that
        callers (and tests) can confirm whether an arbitrary model identifier is
        acceptable without mutating state.
        """

        return self._check_model_info(model_info)

    def _check_temperature(self, temperature: float) -> float:
        """Validates and adjusts the provided temperature value.
        This method ensures that the temperature is a numeric value (either an integer or a float).
        If the temperature is below 0, it issues a warning and sets the temperature to 0.0.
        If the temperature is above 2, it issues a warning and sets the temperature to 2.0.
        Otherwise, it returns the provided temperature as a float.

        Args:
            temperature (float): The temperature value to validate and adjust.

        Returns:
            float: The validated and adjusted temperature value.
        """
        # Ensure that temperature is a number (float or int)
        if not isinstance(temperature, (int, float)):
            raise ValueError("The temperature field must be a float or integer value.")

        # If temperature is below 0, warn and set to 0
        if temperature < 0:
            warnings.warn(
                f"Provided temperature {temperature} is below 0. Setting temperature to 0..."
            )
            return 0.0

        # If temperature is above 2, warn and set to 2
        if temperature > 2:
            warnings.warn(
                f"Provided temperature {temperature} is greater than 2. Setting temperature to 2..."
            )
            return 2.0

        # Otherwise, return the provided temperature as a float
        return float(temperature)

    def _check_demographic_profiles(
        self, demographic_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """Checks to ensure that provided demographic_profiles is not empty and contains a ID column.

        Args:
            demographic_profiles (pd.DataFrame): The demographic profiles to be checked.

        Returns:
            str: The validated demographic_profiles.

        Raises:
            ValueError: If the provided demographic_profiles is an empty dataframe or if it does not contain an ID column.
        """
        if demographic_profiles.empty:
            raise ValueError("demographic_profiles DataFrame cannot be empty.")

        if "ID" not in demographic_profiles.columns:
            raise ValueError(
                "demographic_profiles DataFrame should contain an 'ID' column."
            )

        return demographic_profiles

    def _check_max_conversation_length(self, max_conversation_length: int) -> int:
        """Checks if the provided max_conversation is an integer greater than or equal to 1.

        Args:
            max_conversation_length (int): The max_conversation_length to be checked.

        Returns:
            int: The validated max_conversation_length.

        Raises:
            ValueError: If the provided treatments is less than 1.
        """
        if max_conversation_length < 1:
            raise ValueError(
                "Invalid value for max_conversation_length. Please ensure that max_conversation_length is an integer greater than or equal to 1."
            )

        return max_conversation_length

    def _check_treatments(self, treatments: dict[str, Any]) -> dict[str, Any]:
        """Checks if the provided treatments is valid.

        Args:
            treatments (dict[str, str]): The treatments to be checked.

        Returns:
            dict[str, str]: The validated treatments.

        Raises:
            ValueError: If the provided treatments is not in the correct format.
        """
        for label, treatment_description in treatments.items():
            if not isinstance(treatment_description, str):
                raise ValueError(
                    f"Invalid treatment description: {treatment_description}. Treatment descriptions should be strings."
                )

        return treatments

    def _check_treatment_assignment_strategy(
        self,
        treatment_assignment_strategy: str,
        treatment_column: str,
        session_assignment_strategy: str,
    ) -> str:
        if (
            treatment_assignment_strategy
            not in SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES
        ):
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

        # Check that treatment_column and session_column can be found in demographic_profiles when using manual treatment assignment
        if treatment_assignment_strategy == "manual":
            if (
                treatment_column == ""
                or treatment_column not in self.demographic_profiles.columns
            ):
                raise ValueError(
                    f"The argument 'treatment_column' cannot be an empty string and must be one of the columns in demographic_profiles when using manual treatment assignment."
                )

            if session_assignment_strategy != "manual":
                raise ValueError(
                    f"When using manual treatment assignment, session assignment strategy must also be 'manual' to ensure that subjects in the same session experienced the same treatment arm."
                )

        return treatment_assignment_strategy

    def _check_session_assignment_strategy(
        self, session_assignment_strategy: str, session_column: str
    ) -> str:
        """Checks if the provided session_assignment_strategy is supported.

        Args:
            session_assignment_strategy (str): The session_assignment_strategy to be checked.
            session_column (str): The column name containing the session information when using manual assignment strategy.

        Returns:
            str: The validated session_assignment_strategy.

        Raises:
            ValueError: If the provided session_assignment_strategy is not supported.
            ValueError: If session_column is an empty string or not one of the columns in demographic_profiles when using the manual session assignment strategy.
        """
        if session_assignment_strategy not in SUPPORTED_SESSION_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported session_assignment_strategy: {session_assignment_strategy}. Supported strategies are: {SUPPORTED_SESSION_ASSIGNMENT_STRATEGIES}."
            )

        # Check that session_column can be found in demographic_profiles when using manual session assignment
        if session_assignment_strategy == "manual":
            if (
                session_column == ""
                or session_column not in self.demographic_profiles.columns
            ):
                raise ValueError(
                    f"The argument 'session_column' cannot be an empty string and must be one of the columns in demographic_profiles when performing manual session assignment."
                )

        return session_assignment_strategy

    def _check_role_assignment_strategy(
        self, role_assignment_strategy: str, role_column: str
    ) -> str:
        """Checks if the provided role_assignment_strategy is supported.

        Args:
            role_assignment_strategy (str): The role_assignment_strategy to be checked.
            role_column (str): The column name containing the role information when using manual assignment strategy.

        Returns:
            str: The validated role_assignment_strategy.

        Raises:
            ValueError: If the provided role_assignment_strategy is not supported.
            ValueError: If role_column is an empty string or not one of the columns in demographic_profiles when using the manual role assignment strategy.
        """
        if role_assignment_strategy not in SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported role_assignment_strategy: {role_assignment_strategy}. Supported strategies are: {SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES}."
            )

        # Check that role_column can be found in demographic_profiles when using manual role assignment
        if role_assignment_strategy == "manual":
            if (
                role_column == ""
                or role_column not in self.demographic_profiles.columns
            ):
                raise ValueError(
                    f"The argument 'role_column' cannot be an empty string and must be one of the columns in demographic_profiles when performing manual role assignment."
                )

        return role_assignment_strategy


class AItoAIConversationalExperiment(AIConversationalExperiment):
    """A class representing an AI-to-AI conversational experiment. Inherits from the AIConversationalExperiment class.

    This class extends the `AIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI conversational experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, str]): Dictionary mapping roles to their descriptions.
        num_subjects_per_session (int, optional): Number of subjects per session. Defaults to 2.
        num_sessions (int, optional): Number of sessions. Defaults to 1.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        include_backstories (bool, optional): Whether to include backstories in the subjects profiles. Defaults to False.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided demographic_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_subjects_per_session is less than 2 or will exceed the total number of profile information.
        ValueError: If the provided number of roles is not equal to num_subjects_per_session.
        ValueError: If the number of roles defined does not match the number of subjects assigned to each session.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, str]): The roles assigned to subjects.
        num_subjects_per_session (int): The number of subjects per session.
        num_sessions (int): The number of sessions in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
        include_backstories (bool, optional): Whether to include backstories in the subject's profiles.
        session_id_list (list): A list of session IDs generated based on the number of sessions.
        treatment_assignment (dict[Any, str]): A dictionary mapping session IDs to treatment labels.
        session_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping session IDs to a list of profile information.
        role_assignment (dict[Any, str]): A dictionary mapping user IDs to a specified role.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        demographic_profiles: pd.DataFrame,
        roles: dict[str, str],
        num_subjects_per_session: int = 2,
        num_sessions: int = 1,
        experiment_context: str = "",
        experiment_id: str = "",
        hf_inference_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        include_backstories: bool = False,
    ):
        super().__init__(
            model_info,
            temperature,
            demographic_profiles,
            experiment_context,
            experiment_id,
            hf_inference_endpoint,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            session_assignment_strategy,
            session_column,
            role_assignment_strategy,
            role_column,
            random_seed,
            include_backstories,
        )

        self.roles = roles
        self.num_sessions = self._check_num_sessions(num_sessions=num_sessions)
        self.num_subjects_per_session = self._check_num_subjects_per_session(
            num_subjects_per_session=num_subjects_per_session
        )
        self.session_id_list = self._generate_session_id_list()
        self.treatment_assignment = self._assign_treatment(random_seed=self.random_seed)
        if self.treatment_assignment_strategy == "manual":
            self._check_manually_assigned_treatments()
        self.session_assignment = self._assign_session(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()

    def _check_num_subjects_per_session(self, num_subjects_per_session: int) -> int:
        """Checks if the provided num_subjects_per_session is 2 or more and matches with the number of profiles provided.

        Args:
            num_subjects_per_session (int): The num_subjects_per_session to be checked.

        Returns:
            int: The validated num_subjects_per_session.

        Raises:
            ValueError: If the provided num_subjects_per_session is not valid.
        """
        # Check if number of subjects per session is 2 or more
        if num_subjects_per_session < 2:
            raise ValueError(
                f"Invalid num_subjects_per_session: {num_subjects_per_session}. For AI-AI conversation-based experiments, num_subjects_per_session should be an integer that is equal to or greater than 2."
            )

        # Check if number of subjects per session multipled by the number of sessions is less than the number of profiles provided
        if self.num_sessions * num_subjects_per_session != len(
            self.demographic_profiles
        ):
            raise ValueError(
                f"Total number of subjects required for experiment ({self.num_sessions * num_subjects_per_session}) does not match with the number of profiles provided in demographic_profiles ({len(self.demographic_profiles)})."
            )

        return num_subjects_per_session

    def _check_num_sessions(self, num_sessions: int) -> int:
        """Checks if the provided num_sessions is greater than or equal to 1.

        Args:
            num_sessions (int): The num_sessions to be checked.

        Returns:
            int: The validated num_sessions.

        Raises:
            ValueError: If the provided check_num_sessions is not valid.
        """
        if num_sessions < 1:
            raise ValueError(
                f"Invalid value for num_sessions: {num_sessions}. num_sessions should be an integer that is equal to or greater than 1."
            )

        return num_sessions

    def _generate_session_id_list(self) -> List[Any]:
        """Generates a list of session IDs.

        If the session assignment strategy is set to 'manual',
        the function returns a list of unique session IDs from the demographic_profiles DataFrame.
        Otherwise, it returns a list of sequential integers starting from 0 up to the number of sessions - 1.

        Returns:
            List[Any]: A list of session IDs. If the assignment strategies are manual, the list contains unique session IDs
                from the session_column in the demographic_profiles DataFrame. Otherwise, it contains sequential integers starting from 0.
        """
        if self.session_assignment_strategy == "manual":
            return list(self.demographic_profiles[self.session_column].unique())
        else:
            return list(range(self.num_sessions))

    def _assign_treatment(self, random_seed: int) -> dict[int, str]:
        """Assign treatments to sessions based on the specified treatment assignment strategy.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, str]: A dictionary where the keys represent session numbers and the values represent the assigned treatment labels.
        """
        if self.treatment_assignment_strategy == "simple_random":
            treatment_labels = list(self.treatments.keys())
            return simple_random_assignment_session(
                treatment_labels=treatment_labels,
                session_id_list=self.session_id_list,
                random_seed=random_seed,
            )

        elif self.treatment_assignment_strategy == "complete_random":
            treatment_labels = list(self.treatments.keys())
            return complete_random_assignment_session(
                treatment_labels=treatment_labels,
                session_id_list=self.session_id_list,
                random_seed=random_seed,
            )

        elif self.treatment_assignment_strategy == "manual":
            return manual_assignment_session(
                demographic_profiles=self.demographic_profiles,
                treatment_column=self.treatment_column,
                session_column=self.session_column,
                session_id_list=self.session_id_list,
            )

        else:
            raise ValueError(
                f"Invalid treatment_assignment_strategy: {self.treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

    def _check_manually_assigned_treatments(self) -> None:
        """Checks if the manually defined treatments align with the treatment labels provided in self.treatments.

        Raises:
            ValueError: If the manually defined treatments do not align with the treatment labels provided in self.treatments.
        """
        treatment_label_set = set(self.treatments.keys())
        manual_defined_treatments = set(self.treatment_assignment.values())

        if not treatment_label_set.issuperset(manual_defined_treatments):
            raise ValueError(
                f"The treatment labels defined in the treatments worksheet ({list[treatment_label_set]}) is not a superset of the manually defined treatments in the demographic_profiles worksheet ({list[manual_defined_treatments]})."
            )
        else:
            pass

    def _assign_session(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns profiles to each session based on the given number of subjects per session and session assignment strategy.
        However, if the session_assignment_strategy is 'manual', then assign the subjects to their respective sessions based on the
        assignment defined in demographic_profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping session IDs to a list of profile information.
        """
        if self.session_assignment_strategy == "manual":
            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_subjects = self.demographic_profiles[
                    self.demographic_profiles[self.session_column] == session_id
                ]

                num_session_subjects = len(session_subjects)
                if num_session_subjects != self.num_subjects_per_session:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_subjects} subjects while the number of subjects per session is supposed to be {self.num_subjects_per_session}"
                    )

                session_assignment[session_id] = session_subjects.to_dict(
                    orient="records"
                )

        else:
            randomised_demographic_profiles = self.demographic_profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_assignment[session_id] = randomised_demographic_profiles.iloc[
                    i
                    * self.num_subjects_per_session : (i + 1)
                    * self.num_subjects_per_session
                ].to_dict(orient="records")

        return session_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to subjects based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping subject IDs to their assigned roles.

        Raises:
            ValueError: If the number of defined roles does not match the number of subjects
                        assigned to a session when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.demographic_profiles.set_index("ID")[
                self.role_column
            ].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            role_labels = list(self.roles.keys())
            for session_id, session_subjects in self.session_assignment.items():
                num_subjects = len(session_subjects)

                if len(role_labels) == num_subjects:
                    randomized_roles = random.sample(role_labels, num_subjects)

                else:
                    raise ValueError(
                        f"Number of roles defined ({len(role_labels)}) does not match the number of subjects ({num_subjects}) assigned to Session {session_id}."
                    )

                role_assignment.update(
                    {
                        subject["ID"]: role
                        for subject, role in zip(session_subjects, randomized_roles)
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in the roles worksheet.

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `roles` dictionary.
        If any manually assigned role is not present in the defined roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `roles` worksheet are not a
            superset of the manually defined roles in the `demographic_profiles` worksheet.
        """
        role_label_set = set(self.roles.keys())
        manual_defined_roles = set(self.role_assignment.values())

        if not role_label_set.issuperset(manual_defined_roles):
            raise ValueError(
                f"The roles defined in the roles worksheet ({list[role_label_set]}) is not a superset of the manually defined roles in the demographic_profiles worksheet ({list[manual_defined_roles]})."
            )
        else:
            pass

    def run_experiment(
        self,
        test_mode: bool = True,
        version: int = 1,
        save_results_as_csv: bool = False,
    ) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation.
        If test_mode is set to True, only a random session for each treatment arm will be selected and run sequentially; otherwise, sessions are run in parallel.

        Args:
            test_mode (bool, optional): Indicates whether the experiment is in test mode or not.
                Defaults to True.
            version (int, optional): Indicates the version of the experiment.
                Defaults to 1.
            save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

        Returns:
            dict[str, Any]: A dictionary containing the experiment ID and session information.
        """
        if test_mode:  # Run one session from each treatment group
            session_id_list = []
            for treatment in list(self.treatments.keys()):
                matching_sessions = [
                    sid
                    for sid, assigned in self.treatment_assignment.items()
                    if assigned == treatment
                ]
                if matching_sessions:
                    session_id_list.append(random.choice(matching_sessions))

        else:
            session_id_list = self.session_id_list

        experiment = {
            "experiment_id": f"{self.experiment_id}_{version}",
            "sessions": {},
        }

        # Helper function to process a single session.
        def process_session(session_id: Any) -> tuple[Any, dict]:
            session_info = {}
            session_info["session_id"] = session_id
            session_info["random_seed"] = self.random_seed
            session_info["treatment"] = self.treatments[
                self.treatment_assignment[session_id]
            ]
            session_info["treatment_label"] = self.treatment_assignment[session_id]
            session_info["session_system_message"] = generate_session_system_message(
                experiment_context=self.experiment_context
            )
            session_info["experiment_context"] = self.experiment_context
            session_info["demographic_profiles"] = self.session_assignment[session_id]
            session_subject_ids = [
                profile["ID"] for profile in session_info["demographic_profiles"]
            ]
            session_info["roles"] = {
                subject_id: assigned_role
                for subject_id, assigned_role in self.role_assignment.items()
                if subject_id in session_subject_ids
            }
            session_info["subjects"] = self._initialize_subjects(session_info)
            session_info = self._run_session(session_info, test_mode=test_mode)
            updated_session_subjects = {}
            for subject_role, subject in session_info["subjects"].items():
                updated_session_subjects[subject_role] = subject.to_dict()
            session_info["subjects"] = updated_session_subjects

            return session_id, session_info

        if test_mode:
            # Sequentially process sessions in test mode.
            for session_id in tqdm(session_id_list):
                sid, session_info = process_session(session_id)
                experiment["sessions"][sid] = session_info
        else:
            # Process sessions in parallel using ThreadPoolExecutor.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_sid = {
                    executor.submit(process_session, session_id): session_id
                    for session_id in session_id_list
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_sid),
                    total=len(future_to_sid),
                ):
                    sid, session_info = future.result()
                    experiment["sessions"][sid] = session_info

        self._save_experiment(experiment, save_results_as_csv=save_results_as_csv)

        return experiment

    def _initialize_subjects(
        self, session_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticSubject]:
        """Initializes and returns a dictionary of ConversationalSyntheticSubject objects based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including subjects' profile, role, session ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticSubject]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticSubject objects.

        Raises:
            AssertionError: If the number of demographic profiles does not match the number of roles when initializing subjects.
        """
        assert len(session_info["demographic_profiles"]) == len(
            session_info["roles"]
        ), "Number of demographic profiles does not match the number of roles when initialising subjects."
        subject_dict = {}
        for i in range(len(session_info["demographic_profiles"])):
            subject_id = session_info["demographic_profiles"][i]["ID"]
            role = session_info["roles"][subject_id]
            subject_dict[role] = ConversationalSyntheticSubject(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info=session_info["demographic_profiles"][i],
                model_info=self.model_info,
                temperature=self.temperature,
                include_backstories=self.include_backstories,
                hf_inference_endpoint=self.hf_inference_endpoint,
                role=role,
                role_description=self.roles[role],
                treatment=session_info["treatment"],
            )

        return subject_dict

    def _run_session(
        self, session_info: dict[str, Any], test_mode: bool = False
    ) -> dict[str, Any]:
        """Runs a session involving a conversation between multiple synthetic subjects.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only the first session is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        session_message_history = []
        subject_message_history = {}
        conversation_length = 0
        num_subjects = len(session_info["subjects"])
        subject_list = list(session_info["subjects"].values())
        response = session_info["session_system_message"]
        role = "system"

        while (
            "Thank you for the conversation" not in response
            and conversation_length < self.max_conversation_length
        ):
            if role == "system" and conversation_length == 0:
                message_dict = {
                    role: response,
                    "task_id": conversation_length,
                }
                for subject in subject_list:
                    subject_message_history[subject.role] = [message_dict]

            else:
                message_dict = {
                    role: response,
                    "subject_id": subject_id,
                    "task_id": conversation_length,
                }
                for subject in subject_list:
                    subject_message_history[subject.role].append(message_dict)

            session_message_history.append(message_dict)

            if test_mode:
                print(message_dict)
                print()

            # If no interview script is provided, the sequence of conversation will follow the sequence of subjects defined in self._initialize_subjects
            subject = subject_list[conversation_length % num_subjects]
            subject_id = subject.profile_info.get("ID", "")
            role = subject.role
            response = subject.respond(
                latest_message_history=subject_message_history[role]
            )
            subject_message_history[role] = []
            conversation_length += 1

        message_dict = {
            role: response,
            "subject_id": subject_id,
            "task_id": conversation_length,
        }
        session_message_history.append(message_dict)
        session_message_history.append({"system": "End"})
        if test_mode:
            print(message_dict)
            print()
            print({"system": "End"})

        session_info["message_history"] = session_message_history
        return session_info

    def _save_experiment(
        self, experiment: dict[int, Any], save_results_as_csv: bool = False
    ) -> None:
        """Save the experimental data.

        Args:
            experiment (dict[int, Any]): The experiment data to be saved.
            save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

        Returns:
            None
        """
        save_experiment(experiment, save_results_as_csv)


class AItoAIInterviewExperiment(AItoAIConversationalExperiment):
    """A class representing an AI-to-AI interview experiment. Inherits from the AItoAIConversationalExperiment class.

    This class extends the `AItoAIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI interview experiments.

    Args:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, str]): Dictionary mapping of roles to their descriptions.
        num_subjects_per_session (int, optional): Number of subjects per session. Defaults to 1.
        num_sessions (int, optional): Number of sessions. Defaults to 1.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to subjects. Defaults to "simple_random".
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions. Defaults to "random".
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        include_backstories (bool, optional): Whether to include backstories in the subject profiles. Defaults to False.
        interview_prompts (List[dict[str, str]], optional): An optional dictionary containing the interview script that the facilitator has to follow.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided temperature information is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided demographic_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_subjects_per_session is less than 1 or will exceed the total number of profile information provided.
        ValueError: If the provided number of user-defined roles is not equal to num_subjects_per_session.
        ValueError: If the number of user-defined roles does not match the number of subjects assigned to each session.
        ValueError: If the format of the interview_prompts does not fit with the expected format.

    Attributes:
        model_info (str): The information about the LLM used in the experiment.
        temperature (float): The temperature setting that will be applied to the LLM.
        demographic_profiles (pd.DataFrame): The profile information of the subjects participating in the experiment.
        roles (dict[str, str]): The roles assigned to subjects.
        num_subjects_per_session (int): The number of subjects per session.
        num_sessions (int): The number of sessions in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        hf_inference_endpoint (str, optional): The API inference endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to subjects.
        treatment_column (str, optional): The column in demographic_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        session_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning subjects to sessions.
        role_column (str, optional): The column in demographic_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
        include_backstories (bool, optional): Whether to include backstories in the subject profiles.
        session_id_list (list): A list of session IDs generated based on the number of sessions.
        treatment_assignment (dict[Any, str]): A dictionary mapping session IDs to treatment labels.
        session_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping session IDs to a list of profile information.
        role_assignment (dict[Any, str]): A dictionary mapping user IDs to a specified role.
        interview_prompts (List[dict[str, str]], optional): An optional dictionary containing the interview script that the facilitator has to follow.
    """

    def __init__(
        self,
        model_info: str,
        temperature: float,
        demographic_profiles: pd.DataFrame,
        roles: dict[str, str],
        num_subjects_per_session: int = 1,
        num_sessions: int = 1,
        experiment_context: str = "",
        experiment_id: str = "",
        hf_inference_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        include_backstories: bool = False,
        interview_prompts: List[dict[str, str]] = [],
    ):
        super().__init__(
            model_info,
            temperature,
            demographic_profiles,
            roles,
            num_subjects_per_session,
            num_sessions,
            experiment_context,
            experiment_id,
            hf_inference_endpoint,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            session_assignment_strategy,
            session_column,
            role_assignment_strategy,
            role_column,
            random_seed,
            include_backstories,
        )

        self.roles = self._check_roles(roles=roles)
        self.num_subjects_per_session = self._check_num_subjects_per_session(
            num_subjects_per_session=num_subjects_per_session
        )
        self.session_assignment = self._assign_session(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()
        self.interview_prompts = self._check_prompts(
            interview_prompts=interview_prompts
        )

    def _check_roles(self, roles: dict[str, str]) -> dict[str, str]:
        """Checks if the provided roles are valid.

        Args:
            roles (dict[str, str]): The roles to be checked.

        Returns:
            dict[str, str]: The validated roles.

        Raises:
            ValueError: If the provided roles is not valid.
        """
        if "Facilitator" not in list(roles.keys()):
            raise ValueError(
                "For an AI-to-AI interview-based experiment, one of the roles must be 'Facilitator'."
            )

        return roles

    def _check_num_subjects_per_session(self, num_subjects_per_session: int) -> int:
        """Checks if the provided num_subjects_per_session is 1 or more and matches with the number of demographic profiles provided.

        Args:
            num_subjects_per_session (int): The num_subjects_per_session to be checked.

        Returns:
            int: The validated num_subjects_per_session.

        Raises:
            ValueError: If the provided num_subjects_per_session is not valid.
        """
        # Ensure that number of subjects per session is 1 or more
        if num_subjects_per_session < 1:
            raise ValueError(
                f"Invalid num_subjects_per_session: {num_subjects_per_session}. For AI-AI interview-based experiments, num_subjects_per_session should be an integer that is equal to or greater than 1."
            )

        # Ensure that number of subjects per session matches with the number of demographic profiles provided
        user_defined_roles = [
            role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
        ]
        if len(user_defined_roles) != num_subjects_per_session:
            raise ValueError(
                f"Number of user-defined roles ({len(user_defined_roles)}) does not match the number of subjects assigned to each session ({num_subjects_per_session})."
            )

        # Ensure that number of user-defined roles multiplied by the number of sessions is less than or equal to the number of profiles provided
        if self.num_sessions * len(user_defined_roles) > len(self.demographic_profiles):
            raise ValueError(
                f"Total number of subjects required for experiment ({self.num_sessions * len(user_defined_roles)}) larger than the number of profiles provided in demographic_profiles ({len(self.demographic_profiles)})."
            )

        return num_subjects_per_session

    def _assign_session(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns demographic profiles to each session based on the given number of subjects per session (excluding the special roles) and session assignment strategy.
        However, if the session_assignment_strategy is 'manual', then assign the subjects to their respective sessions based on the
        assignment defined in demographic_profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping session IDs to a list of demographic profile information.
        """
        num_user_defined_roles = len(
            [role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES]
        )

        if self.session_assignment_strategy == "manual":
            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_participants = self.demographic_profiles[
                    self.demographic_profiles[self.session_column] == session_id
                ].reset_index(drop=True)

                num_session_participants = len(session_participants)
                if num_session_participants != num_user_defined_roles:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_participants} participants while the number of user-defined roles per session is supposed to be {num_user_defined_roles}"
                    )

                session_assignment[session_id] = session_participants.to_dict(
                    orient="records"
                )

        else:
            randomised_demographic_profiles = self.demographic_profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_assignment[session_id] = randomised_demographic_profiles.iloc[
                    i * num_user_defined_roles : (i + 1) * num_user_defined_roles
                ].to_dict(orient="records")

        return session_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to subjects based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping subject IDs to their assigned roles.

        Raises:
            ValueError: If the number of defined roles does not match the number of subjects
                        assigned to a session when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.demographic_profiles.set_index("ID")[
                self.role_column
            ].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            user_defined_role_labels = [
                role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
            ]
            for session_id, session_subjects in self.session_assignment.items():
                num_subjects = len(session_subjects)

                if len(user_defined_role_labels) == num_subjects:
                    randomized_roles = random.sample(
                        user_defined_role_labels, num_subjects
                    )

                else:
                    raise ValueError(
                        f"Number of user-defined roles ({len(user_defined_role_labels)}) does not match the number of subjects ({num_subjects}) assigned to Session {session_id}."
                    )

                role_assignment.update(
                    {
                        participant["ID"]: role
                        for participant, role in zip(session_subjects, randomized_roles)
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in roles, excluding special roles like "Facilitator" and "Summarizer".

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `roles` dictionary.
        If any manually assigned role is not present in the defined roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `roles` worksheet are not a
            superset of the manually defined roles in the `demographic_profiles` worksheet.
        """
        role_label_set = set(
            [role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES]
        )
        manual_defined_roles = set(self.role_assignment.values())

        if not role_label_set.issuperset(manual_defined_roles):
            raise ValueError(
                f"The user-defined roles in the roles worksheet ({list[role_label_set]}) is not a superset of the manually defined roles in the demographic_profiles worksheet ({list[manual_defined_roles]})."
            )
        else:
            pass

    def _check_prompts(
        self, interview_prompts: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Validates and processes a list of interview prompts.

        Args:
            interview_prompts (List[dict[str, Any]]): A list of dictionaries where each dictionary represents a prompt
                with various attributes such as "type", "var_name", "randomize_response_order", etc.
        Returns:
            List[dict[str, Any]]: The validated and processed list of interview prompts.

        Raises:
            ValueError: If any of the following conditions are not met:
                - The `interview_prompts` list is not empty.
                - The first item in `interview_prompts` contains a "type" field with the value "context".
                - The total length of the interview does not exceed the maximum conversation length
                  (calculated as `len(interview_prompts) * total number of special and user-defined roles).
                - Each prompt's "type" field contains only approved prompt types (defined in `SUPPORTED_PROMPT_TYPES`).
                - Each prompt's "var_name" field contains unique variable names.
                - The "randomize_response_order" field contains only approved values True or False).
                - The "validate_response" field contains only approved values (True or False).
                - The "generate_speculation_score" field contains only approved values (True or False).
                - The "format_response" field contains only approved values (True or False).
        """
        # Check if interview_prompts is not an empty list
        if not interview_prompts:
            raise ValueError("The interview_prompts list should not be an empty list.")

        # Check if the first item in interview_prompts contains "type": "context"
        if (
            "type" not in interview_prompts[0]
            or interview_prompts[0]["type"] != "context"
        ):
            raise ValueError(
                'The first item in interview_prompts must contain "type": "context" to provide context for the experiment/interview.'
            )

        # Check if the length of the interview would exceed the maximum conversation length
        if len(interview_prompts) * len(self.roles) > self.max_conversation_length:
            raise ValueError(
                f"Based on the length of the interview script ({len(interview_prompts)}) and total number of special and user-defined roles ({len(self.roles)}), the maximum length of the conversation should be larger or equal to {len(interview_prompts) * len(self.roles)}, rather than {self.max_conversation_length}."
            )

        unique_var_names = []
        for prompt_dict in interview_prompts:
            # Check if the type field contains only approved prompt types
            if prompt_dict["type"] not in SUPPORTED_PROMPT_TYPES:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid prompt type: {prompt_dict['type']}. Supported prompt types include: {SUPPORTED_PROMPT_TYPES}."
                )

            # Check if the var_name column contains unique variable names
            if prompt_dict["var_name"] in unique_var_names:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains a non-unique variable name: {prompt_dict['var_name']}."
                )
            else:
                unique_var_names.append(prompt_dict["var_name"])

            # Check if the randomize_response_order column contains only approved values (True or False)
            if prompt_dict["randomize_response_order"] not in [True, False]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in randomize_response_order field: {prompt_dict['randomize_response_order']}. Supported options include: True or False."
                )

            # Check if the validate_response column contains only approved values (True or False)
            if prompt_dict["validate_response"] not in [True, False]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in validate_response field: {prompt_dict['validate_response']}. Supported options include: True or False."
                )

            # Check if the generate_speculation_score column contains only approved values (True or False)
            if prompt_dict["generate_speculation_score"] not in [True, False]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in generate_speculation_score field: {prompt_dict['generate_speculation_score']}. Supported options include: True or False."
                )

            # Check if the format_response column contains only approved values (True or False)
            if prompt_dict["format_response"] not in [True, False]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in format_response field: {prompt_dict['format_response']}. Supported options include: True or False."
                )

        return interview_prompts

    def run_experiment(
        self,
        test_mode: bool = True,
        version: int = 1,
        save_results_as_csv: bool = False,
    ) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation.
        If test_mode is set to True, a random session from each treatment arm will be selected and run; otherwise, sessions are run in parallel.

        Args:
            test_mode (bool, optional): Indicates whether the experiment is in test mode or not.
                Defaults to True.
            version (int, optional): Indicates the version of the experiment.
                Defaults to 1.
            save_results_as_csv (bool, optional): Indicates whether the results of the experiment will be saved as CSV format.
                Defaults to False

        Returns:
            dict[str, Any]: A dictionary containing the experiment ID and session information.
        """
        if test_mode:  # Run one session of each of the treatment groups
            random.seed(self.random_seed)
            session_id_list = []
            for treatment in list(self.treatments.keys()):
                matching_sessions = [
                    sid
                    for sid, assigned in self.treatment_assignment.items()
                    if assigned == treatment
                ]
                if matching_sessions:
                    session_id_list.append(random.choice(matching_sessions))

        else:
            session_id_list = self.session_id_list

        experiment = {
            "experiment_id": f"{self.experiment_id}_{version}",
            "sessions": {},
        }
        self.experiment_context = self.interview_prompts.pop(0)["llm_text"][
            "Facilitator"
        ]

        # Helper function to process a single session.
        def process_session(session_id: Any) -> tuple[Any, dict]:
            session_info = {}
            session_info["session_id"] = session_id
            session_info["random_seed"] = self.random_seed
            session_info["treatment"] = self.treatments[
                self.treatment_assignment[session_id]
            ]
            session_info["treatment_label"] = self.treatment_assignment[session_id]
            session_info["session_system_message"] = generate_session_system_message(
                experiment_context=self.experiment_context,
            )
            session_info["experiment_context"] = self.experiment_context
            session_info["demographic_profiles"] = self.session_assignment[session_id]
            session_subject_ids = [
                profile["ID"] for profile in session_info["demographic_profiles"]
            ]
            session_info["roles"] = {
                subject_id: assigned_role
                for subject_id, assigned_role in self.role_assignment.items()
                if subject_id in session_subject_ids
            }
            session_info["subjects"] = self._initialize_subjects(session_info)
            session_info = self._run_session(
                session_info=session_info,
                interview_prompts=self.interview_prompts,
                test_mode=test_mode,
            )
            updated_session_subjects = {}
            for role, subject in session_info["subjects"].items():
                updated_session_subjects[role] = subject.to_dict()
            session_info["subjects"] = updated_session_subjects

            return session_id, session_info

        if test_mode:
            # Sequentially process sessions in test mode.
            for session_id in tqdm(session_id_list):
                sid, session_info = process_session(session_id)
                experiment["sessions"][sid] = session_info
        else:
            # Process sessions in parallel using ThreadPoolExecutor.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_sid = {
                    executor.submit(process_session, session_id): session_id
                    for session_id in session_id_list
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_sid),
                    total=len(future_to_sid),
                ):
                    sid, session_info = future.result()
                    experiment["sessions"][sid] = session_info

        self._save_experiment(experiment, save_results_as_csv=save_results_as_csv)

        return experiment

    def _initialize_subjects(
        self, session_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticSubject]:
        """Initializes and returns a dictionary of ConversationalSyntheticSubject objects for both special and user-defined roles based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including demographic profiles, roles, session ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticSubject]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticSubject objects.

        Raises:
            AssertionError: If the number of demographic profiles does not match the number of user-defined roles when initializing subjects.
        """
        subject_dict = {}

        # First, initialise Facilitator
        subject_dict["Facilitator"] = ConversationalSyntheticSubject(
            experiment_id=self.experiment_id,
            experiment_context=self.experiment_context,
            session_id=session_info["session_id"],
            profile_info={},
            model_info=self.model_info,
            temperature=self.temperature,
            hf_inference_endpoint=self.hf_inference_endpoint,
            role="Facilitator",
            role_description=self.roles["Facilitator"],
            treatment="",
            include_backstories=False,
        )

        # Initialise user-defined subjects based on sequence defined in roles
        user_defined_roles = [
            role for role in list(self.roles.keys()) if role not in SPECIAL_ROLES
        ]
        assert len(session_info["demographic_profiles"]) == len(
            user_defined_roles
        ), f"Number of demographic profiles ({len(session_info['demographic_profiles'])}) does not match the number of user-defined roles ({len(user_defined_roles)}) when initialising subjects. The number of demographic profiles should be equal the number of user-defined roles (excluding special roles like Facilitator and Summarizer)."

        for i in range(len(session_info["demographic_profiles"])):
            subject_id = session_info["demographic_profiles"][i]["ID"]
            role = session_info["roles"][subject_id]
            subject_dict[role] = ConversationalSyntheticSubject(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info=session_info["demographic_profiles"][i],
                model_info=self.model_info,
                temperature=self.temperature,
                hf_inference_endpoint=self.hf_inference_endpoint,
                role=role,
                role_description=self.roles[role],
                treatment=session_info["treatment"],
                include_backstories=self.include_backstories,
            )

        # Define Summarizer last if it is required
        if "Summarizer" in list(self.roles.keys()):
            subject_dict["Summarizer"] = ConversationalSyntheticSubject(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info={},
                model_info=self.model_info,
                temperature=self.temperature,
                hf_inference_endpoint=self.hf_inference_endpoint,
                role="Summarizer",
                role_description=self.roles["Summarizer"],
                treatment="",
                include_backstories=False,
            )

        return subject_dict

    def _sort_tasks(self, prompts: list[dict]) -> list[dict]:
        """Sorts and shuffles a list of prompts based on their "task_order" value.
        This method groups the input prompts by their "task_order" value, sorts the groups
        in ascending order of "task_order", and shuffles the prompts within each group
        randomly. The shuffled groups are then concatenated to form the final sorted list.

        Args:
            prompts (list[dict]): A list of dictionaries, where each dictionary represents
                a prompt and contains a "task_order" key.

        Returns:
            list[dict]: A list of prompts sorted by "task_order" and shuffled within each
                "task_order" group.
        """
        random.seed(self.random_seed)

        # Group prompts by their task_order value.
        groups = defaultdict(list)
        for prompt in prompts:
            groups[prompt["task_order"]].append(prompt)

        # For prompts with the same task_order, shuffle the group randomly.
        sorted_prompts = []
        for order in sorted(groups.keys()):  # Ascending order on task_order
            group = groups[order]
            random.shuffle(group)  # Randomly shuffle prompts with the same task_order
            sorted_prompts.extend(group)

        return sorted_prompts

    def _format_response_options(
        self, response_options: Any, randomize_response_order: bool = False
    ) -> str:
        """Formats the given response options into a human-readable string.

        Args:
            response_options (Any): The response options to format. This can be a range,
                                    a list, or any other type.
            randomize_response_order (bool, optional): Indicates whether the response options should be randomized.
                Defaults to False.

        Returns:
            str: A formatted string representation of the response options.
                 - If `response_options` is a range, it is formatted as "from X to Y".
                 - If `response_options` is a list, its elements are joined with commas.
                 - For other types, it falls back to a simple string conversion.
        """
        random.seed(self.random_seed)
        # If response_options is a range, format it as "from X to Y"
        if isinstance(response_options, range):
            return f"Respond with a numerical value ranging from {response_options.start} to {response_options.stop} (inclusive):"

        # If it's a list, join the elements with commas.
        elif isinstance(response_options, list):
            if len(response_options) == 0:
                formatted = ""
            elif len(response_options) == 1:
                formatted = str(response_options[0])
            else:
                if randomize_response_order:
                    random.shuffle(response_options)
                formatted = (
                    ", ".join(f"'{opt}'" for opt in response_options[:-1])
                    + f" or '{response_options[-1]}'"
                )
            return f"Respond with one option from {formatted}:"

        # Otherwise, fallback to a simple string conversion.
        else:
            return str(response_options)

    def _run_session(
        self,
        session_info: dict[str, Any],
        interview_prompts: List[dict[str, str]],
        test_mode: bool = False,
    ) -> dict[str, Any]:
        """Runs a session involving an interview between multiple synthetic subjects.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            interview_prompts (List[dict[str, str]]): An list containing the interview script that the facilitator has to follow.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only a randomly seleced session from each treatment arm is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        session_message_history = []
        subject_message_history = {}
        response = session_info["session_system_message"]
        role = "system"
        message_dict = {role: response}

        subject_list = list(session_info["subjects"].values())
        for subject in subject_list:
            subject_message_history[subject.role] = [message_dict]
        session_message_history.append(message_dict)

        if test_mode:
            print(message_dict)
            print()

        # Sort the order of tasks based on the task_order field. If task order is repeated, then it is expected that the task order are randomised
        interview_prompts = self._sort_tasks(interview_prompts)

        for round in interview_prompts:
            # Facilitator is providing instructions at the beginning to all subjects and allowing the subjects to continue the discussion.
            if round["type"] in ["context", "discussion"]:
                # Format context/discussion question
                response = round["llm_text"]["Facilitator"]
                if "{response_options}" in response:
                    try:
                        response = response.replace(
                            "{response_options}",
                            self._format_response_options(
                                response_options=round["response_options"][
                                    "Facilitator"
                                ],
                                randomize_response_order=round[
                                    "randomize_response_order"
                                ],
                            ),
                        )
                    except KeyError:
                        raise KeyError(
                            f"The role 'Facilitator' was not found in the response_options dictionary for task ID {round.get('task_id', None)}."
                        )

                message_dict = {
                    "Facilitator": response,
                    "task_id": round.get("task_id", None),
                    "var_name": round.get("var_name", None),
                }

                for subject in subject_list:
                    if subject.role == "Facilitator":
                        continue
                    subject_message_history[subject.role].append(message_dict)
                session_message_history.append(message_dict)

                if test_mode:
                    print(message_dict)
                    print()

                if (
                    round["type"] == "context"
                ):  # Context setting only, no response required from subjects
                    continue

                # Loop through each subject back-to-back and get their response during a discussion round
                for role, subject in session_info["subjects"].items():
                    if role == "Facilitator":
                        continue
                    response = subject.respond(
                        latest_message_history=subject_message_history[role],
                        generate_speculation_score=round["generate_speculation_score"],
                        format_response=round["format_response"],
                    )
                    subject_message_history[role] = []

                    message_dict = {
                        role: response,
                        "subject_id": subject.profile_info.get("ID", ""),
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    for subject in subject_list:
                        subject_message_history[subject.role].append(message_dict)
                    session_message_history.append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

            elif round["type"] == "public_question":
                # Facilitator is posing the same question to each subject and the subjects' responses are shown to all subjects during the round.
                for role, question in round["llm_text"].items():
                    # Format interview question
                    if "{response_options}" in question:
                        try:
                            question = question.replace(
                                "{response_options}",
                                self._format_response_options(
                                    response_options=round["response_options"][role],
                                    randomize_response_order=round[
                                        "randomize_response_order"
                                    ],
                                ),
                            )
                        except KeyError:
                            raise ValueError(
                                f"KeyError: The role '{role}' was not found in the response_options dictionary for task ID {round.get('task_id', None)}."
                            )

                    message_dict = {
                        "Facilitator": question,
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    for subject in subject_list:
                        subject_message_history[subject.role].append(message_dict)
                    session_message_history.append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

                    subject = session_info["subjects"][role]
                    response = subject.respond(
                        latest_message_history=subject_message_history[role],
                        validate_response=round["validate_response"],
                        response_options=round["response_options"].get(role, []),
                        generate_speculation_score=round["generate_speculation_score"],
                        format_response=round["format_response"],
                    )
                    subject_message_history[role] = []

                    message_dict = {
                        role: response,
                        "subject_id": subject.profile_info.get("ID", ""),
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    for subject in subject_list:
                        subject_message_history[subject.role].append(message_dict)
                    session_message_history.append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

            elif round["type"] == "private_question":
                # Facilitator is posing the same question to each subject but the subjects' responses are not shown to other subjects during the round
                for role, question in round["llm_text"].items():
                    # Format interview question
                    if "{response_options}" in question:
                        try:
                            question = question.replace(
                                "{response_options}",
                                self._format_response_options(
                                    response_options=round["response_options"][role],
                                    randomize_response_order=round[
                                        "randomize_response_order"
                                    ],
                                ),
                            )
                        except KeyError:
                            raise ValueError(
                                f"KeyError: The role '{role}' was not found in the response_options dictionary for task ID {round.get('task_id', None)}."
                            )

                    message_dict = {
                        "Facilitator": question,
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    subject_message_history[role].append(message_dict)
                    session_message_history.append(message_dict)
                    if "Summarizer" in subject_message_history and role != "Summarizer":
                        subject_message_history["Summarizer"].append(message_dict)

                    if test_mode:
                        print(message_dict)
                        print()

                    subject = session_info["subjects"][role]
                    response = subject.respond(
                        latest_message_history=subject_message_history[role],
                        validate_response=round["validate_response"],
                        response_options=round["response_options"].get(role, []),
                        generate_speculation_score=round["generate_speculation_score"],
                        format_response=round["format_response"],
                    )
                    subject_message_history[role] = []

                    message_dict = {
                        role: response,
                        "subject_id": subject.profile_info.get("ID", ""),
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    subject_message_history[role].append(message_dict)
                    session_message_history.append(message_dict)
                    if "Summarizer" in subject_message_history and role != "Summarizer":
                        subject_message_history["Summarizer"].append(message_dict)

                    elif (
                        "Summarizer" in subject_message_history and role == "Summarizer"
                    ):
                        for subject in subject_list:
                            if subject.role != "Summarizer":
                                subject_message_history[subject.role].append(
                                    message_dict
                                )

                    else:  # No Summarizer in the session
                        pass

                    if test_mode:
                        print(message_dict)
                        print()

            else:
                raise ValueError(
                    f"Invalid prompt type: {round['type']}. The type of prompt for each interview round should one of these options: {SUPPORTED_PROMPT_TYPES}"
                )

        session_message_history.append({"system": "End"})
        if test_mode:
            print({"system": "End"})

        session_info["message_history"] = session_message_history
        return session_info
