import pandas as pd
import datetime
import random
import warnings
import concurrent.futures
from collections import defaultdict
from typing import Any, List
from tqdm import tqdm
from talkingtomachines.generative.synthetic_agent import (
    ConversationalSyntheticAgent,
    ProfileInfo,
)
from talkingtomachines.management.treatment import (
    simple_random_assignment_session,
    complete_random_assignment_session,
    manual_assignment_session,
)
from talkingtomachines.generative.prompt import (
    generate_conversational_session_system_message,
)
from talkingtomachines.storage.experiment import save_experiment

SUPPORTED_MODELS = [
    "gpt-4.5-preview",
    "o3",
    "o4-mini",
    "o1-pro",
    "o1",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "hf-inference",
]
SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES = [
    "simple_random",
    "complete_random",
    # "full_factorial",
    # "block_randomisation",
    # "cluster_randomisation",
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
    "question",
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
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents. Defaults to "simple_random".
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        role_column (str, optional): The column in agent_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided agent_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to agents.
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        role_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
    """

    def __init__(
        self,
        model_info: str,
        agent_profiles: pd.DataFrame,
        experiment_context: str = "",
        experiment_id: str = "",
        api_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
    ):
        super().__init__(
            experiment_id,
        )

        self.model_info = self._check_model_info(model_info=model_info)
        self.experiment_context = experiment_context
        self.agent_profiles = self._check_agent_profiles(agent_profiles=agent_profiles)
        self.api_endpoint = api_endpoint
        self.max_conversation_length = self._check_max_conversation_length(
            max_conversation_length=max_conversation_length
        )
        self.treatments = self._check_treatments(treatments=treatments)
        self.treatment_assignment_strategy = self._check_treatment_assignment_strategy(
            treatment_assignment_strategy=treatment_assignment_strategy,
            treatment_column=treatment_column,
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

    def _check_model_info(self, model_info: str) -> str:
        """Checks if the provided model_info is supported.

        Args:
            model_info (str): The model_info to be checked.

        Returns:
            str: The validated model_info.

        Raises:
            ValueError: If the provided model_info is not supported based on SUPPORTED_MODELS.
        """
        if model_info not in SUPPORTED_MODELS:
            warnings.warn(
                f"{model_info} is not one of the supported models ({SUPPORTED_MODELS}). Defaulting to querying OpenAI endpoint."
            )

        return model_info

    def _check_agent_profiles(self, agent_profiles: pd.DataFrame) -> pd.DataFrame:
        """Checks to ensure that provided agent_profiles is not empty and contains a ID column.

        Args:
            agent_profiles (pd.DataFrame): The agent_profiles to be checked.

        Returns:
            str: The validated agent_profiles.

        Raises:
            ValueError: If the provided agent_profiles is an empty dataframe or if it does not contain an ID column.
        """
        if agent_profiles.empty:
            raise ValueError("agent_profiles DataFrame cannot be empty.")

        if "ID" not in agent_profiles.columns:
            raise ValueError("agent_profiles DataFrame should contain an 'ID' column.")

        return agent_profiles

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
    ) -> str:
        """Checks if the provided treatment_assignment_strategy is supported.

        Args:
            treatment_assignment_strategy (str): The treatment_assignment_strategy to be checked.
            treatment_column (str): The column name containing information about the manually assigned treatments.

        Returns:
            str: The validated treatment_assignment_strategy.

        Raises:
            ValueError: If the provided treatment_assignment_strategy is not supported.
            ValueError: If treatment_column is an empty string or not one of the columns in agent_profiles when using the manual treatment assignment strategy.
        """
        if (
            treatment_assignment_strategy
            not in SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES
        ):
            raise ValueError(
                f"Unsupported treatment_assignment_strategy: {treatment_assignment_strategy}. Supported strategies are: {SUPPORTED_TREATMENT_ASSIGNMENT_STRATEGIES}."
            )

        # Check that treatment_column and session_column can be found in agent_profiles when using manual treatment assignment
        if treatment_assignment_strategy == "manual":
            if (
                treatment_column == ""
                or treatment_column not in self.agent_profiles.columns
            ):
                raise ValueError(
                    f"The argument 'treatment_column' cannot be an empty string and must be one of the columns in agent_profiles when using manual treatment assignment."
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
            ValueError: If session_column is an empty string or not one of the columns in agent_profiles when using the manual session assignment strategy.
        """
        if session_assignment_strategy not in SUPPORTED_SESSION_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported session_assignment_strategy: {session_assignment_strategy}. Supported strategies are: {SUPPORTED_SESSION_ASSIGNMENT_STRATEGIES}."
            )

        # Check that session_column can be found in agent_profiles when using manual session assignment
        if session_assignment_strategy == "manual":
            if (
                session_column == ""
                or session_column not in self.agent_profiles.columns
            ):
                raise ValueError(
                    f"The argument 'session_column' cannot be an empty string and must be one of the columns in agent_profiles when performing manual session assignment."
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
            ValueError: If role_column is an empty string or not one of the columns in agent_profiles when using the manual role assignment strategy.
        """
        if role_assignment_strategy not in SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES:
            raise ValueError(
                f"Unsupported role_assignment_strategy: {role_assignment_strategy}. Supported strategies are: {SUPPORTED_ROLE_ASSIGNMENT_STRATEGIES}."
            )

        # Check that role_column can be found in agent_profiles when using manual role assignment
        if role_assignment_strategy == "manual":
            if role_column == "" or role_column not in self.agent_profiles.columns:
                raise ValueError(
                    f"The argument 'role_column' cannot be an empty string and must be one of the columns in agent_profiles when performing manual role assignment."
                )

        return role_assignment_strategy


class AItoAIConversationalExperiment(AIConversationalExperiment):
    """A class representing an AI-to-AI conversational experiment. Inherits from the AIConversationalExperiment class.

    This class extends the `AIConversationalExperiment` class and provides additional functionality
    specific to AI-to-AI conversational experiments.

    Args:
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        agent_roles (dict[str, str]): Dictionary mapping agent roles to their descriptions.
        num_agents_per_session (int, optional): Number of agents per session. Defaults to 2.
        num_sessions (int, optional): Number of sessions. Defaults to 10.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents. Defaults to "simple_random".
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        role_column (str, optional): The column in agent_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided agent_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_agents_per_session is less than 2 or will exceed the total number of profile information.
        ValueError: If the provided number of agent_roles is not equal to num_agents_per_session.
        ValueError: If the number of roles defined does not match the number of agents assigned to each session.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        agent_roles (dict[str, str]): The roles assigned to agents.
        num_agents_per_session (int): The number of agents per session.
        num_sessions (int): The number of sessions in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to agents.
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        role_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
        session_id_list (list): A list of session IDs generated based on the number of sessions.
        treatment_assignment (dict[Any, str]): A dictionary mapping session IDs to treatment labels.
        session_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping session IDs to a list of agent profile information.
        role_assignment (dict[Any, str]): A dictionary mapping user IDs to a specified role.
    """

    def __init__(
        self,
        model_info: str,
        agent_profiles: pd.DataFrame,
        agent_roles: dict[str, str],
        num_agents_per_session: int = 2,
        num_sessions: int = 10,
        experiment_context: str = "",
        experiment_id: str = "",
        api_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
    ):
        super().__init__(
            model_info,
            agent_profiles,
            experiment_context,
            experiment_id,
            api_endpoint,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            session_assignment_strategy,
            session_column,
            role_assignment_strategy,
            role_column,
            random_seed,
        )

        self.num_agents_per_session = self._check_num_agents_per_session(
            num_agents_per_session=num_agents_per_session
        )
        self.agent_roles = self._check_agent_roles(agent_roles=agent_roles)
        self.num_sessions = self._check_num_sessions(num_sessions=num_sessions)
        self.session_id_list = self._generate_session_id_list()
        self.treatment_assignment = self._assign_treatment(random_seed=self.random_seed)
        if self.treatment_assignment_strategy == "manual":
            self._check_manually_assigned_treatments()
        self.session_assignment = self._assign_session(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()

    def _check_num_agents_per_session(self, num_agents_per_session: int) -> int:
        """Checks if the provided num_agents_per_session is 2 or more and matches with the number of agent profiles provided.

        Args:
            num_agents_per_session (int): The num_agents_per_session to be checked.

        Returns:
            int: The validated num_agents_per_session.

        Raises:
            ValueError: If the provided num_agents_per_session is not valid.
        """
        if num_agents_per_session < 2:
            raise ValueError(
                f"Invalid num_agents_per_session: {num_agents_per_session}. For AI-AI conversation-based experiments, num_agents_per_session should be an integer that is equal to or greater than 2."
            )

        if self.num_sessions * num_agents_per_session != len(self.agent_profiles):
            raise ValueError(
                f"Total number of agents required for experiment ({self.num_sessions * num_agents_per_session}) does not match with the number of profiles provided in agent_profiles ({len(self.agent_profiles)})."
            )

        return num_agents_per_session

    def _check_agent_roles(self, agent_roles: dict[str, str]) -> dict[str, str]:
        """Checks if the provided agent_roles is valid.

        Args:
            agent_roles (dict[str, str]): The agent_roles to be checked.

        Returns:
            dict[str, str]: The validated agent_roles.

        Raises:
            ValueError: If the provided agent_roles is not valid.
        """
        if len(agent_roles) != self.num_agents_per_session:
            raise ValueError(
                f"Number of roles defined ({len(agent_roles)}) does not match the number of agents assigned to each session ({self.num_agents_per_session})."
            )

        return agent_roles

    def _check_num_sessions(self, num_sessions: int) -> int:
        """Checks if the provided num_sessions is greater than 1.

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

        If either the treatment assignment strategy or the agent assignment strategy is set to 'manual',
        the function returns a list of unique session IDs from the agent profiles DataFrame.
        Otherwise, it returns a list of sequential integers starting from 0 up to the number of sessions - 1.

        Returns:
            List[Any]: A list of session IDs. If the assignment strategies are manual, the list contains unique session IDs
                from the session_column in the agent_profiles DataFrame. Otherwise, it contains sequential integers starting from 0.
        """
        if self.session_assignment_strategy == "manual":
            return list(self.agent_profiles[self.session_column].unique())
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

        # elif self.treatment_assignment_strategy == "full_factorial":
        #     treatment_labels = []
        #     for _, inner_treatment_dict in self.treatments.items():
        #         inner_treatment_labels = list(inner_treatment_dict.keys())
        #         treatment_labels.append(inner_treatment_labels)
        #     return full_factorial_assignment_session(
        #         treatment_labels=treatment_labels, session_id_list=self.session_id_list, random_seed=random_seed
        #     )

        elif self.treatment_assignment_strategy == "manual":
            return manual_assignment_session(
                agent_profiles=self.agent_profiles,
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
                f"The treatment labels defined in the treatments worksheet ({list[treatment_label_set]}) is not a superset of the manually defined treatments in the agent_profiles worksheet ({list[manual_defined_treatments]})."
            )
        else:
            pass

    def _assign_session(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns agent profiles to each session based on the given number of agents per session and agent assignment strategy.
        However, if the session_assignment_strategy is 'manual', then assign the agents to their respective sessions based on the
        assignment defined in agent_profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping session IDs to a list of agent profile information.
        """
        if self.session_assignment_strategy == "manual":
            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_participants = self.agent_profiles[
                    self.agent_profiles[self.session_column] == session_id
                ]

                num_session_participants = len(session_participants)
                if num_session_participants != self.num_agents_per_session:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_participants} participants while the number of participants per session is supposed to be {self.num_agents_per_session}"
                    )

                session_assignment[session_id] = session_participants.to_dict(
                    orient="records"
                )

        else:
            randomised_agent_profiles = self.agent_profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_assignment[session_id] = randomised_agent_profiles.iloc[
                    i
                    * self.num_agents_per_session : (i + 1)
                    * self.num_agents_per_session
                ].to_dict(orient="records")

        return session_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to agents based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping agent IDs to their assigned roles.

        Raises:
            ValueError: If the number of defined roles does not match the number of agents
                        assigned to a session when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.agent_profiles.set_index("ID")[
                self.role_column
            ].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            agent_role_labels = list(self.agent_roles.keys())
            for session_id, session_participants in self.session_assignment.items():
                num_participants = len(session_participants)

                if len(agent_role_labels) == num_participants:
                    randomized_roles = random.sample(
                        agent_role_labels, num_participants
                    )

                else:
                    raise ValueError(
                        f"Number of roles defined ({len(agent_role_labels)}) does not match the number of agents ({num_participants}) assigned to Session {session_id}."
                    )

                role_assignment.update(
                    {
                        participant["ID"]: role
                        for participant, role in zip(
                            session_participants, randomized_roles
                        )
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in the agent roles.

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `agent_roles` dictionary.
        If any manually assigned role is not present in the defined agent roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `agent_roles` worksheet are not a
            superset of the manually defined roles in the `agent_profiles` worksheet.
        """
        role_label_set = set(self.agent_roles.keys())
        manual_defined_roles = set(self.role_assignment.values())

        if not role_label_set.issuperset(manual_defined_roles):
            raise ValueError(
                f"The roles defined in the agent_roles worksheet ({list[role_label_set]}) is not a superset of the manually defined roles in the agent_profiles worksheet ({list[manual_defined_roles]})."
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
        If test_mode is set to True, only the first session will be selected and run; otherwise, sessions are run in parallel.

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
            session_info["treatment"] = self.treatments[
                self.treatment_assignment[session_id]
            ]
            session_info["session_system_message"] = (
                generate_conversational_session_system_message(
                    experiment_context=self.experiment_context,
                    treatment=session_info["treatment"],
                )
            )
            session_info["agent_profiles"] = self.session_assignment[session_id]
            session_participant_ids = [
                profile["ID"] for profile in session_info["agent_profiles"]
            ]
            session_info["roles"] = {
                participant_id: assigned_role
                for participant_id, assigned_role in self.role_assignment.items()
                if participant_id in session_participant_ids
            }
            session_info["agents"] = self._initialize_agents(session_info)
            session_info = self._run_session(session_info, test_mode=test_mode)
            updated_session_agent = {}
            for agent_role, agent in session_info["agents"].items():
                updated_session_agent[agent_role] = agent.to_dict()
            session_info["agents"] = updated_session_agent

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

    def _initialize_agents(
        self, session_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticAgent]:
        """Initializes and returns a dictionary of ConversationalSyntheticAgent objects based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including agents' profile, role, session ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticAgent]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticAgent objects.

        Raises:
            AssertionError: If the number of agent profiles does not match the number of agent roles when initializing agents.
        """
        assert len(session_info["agent_profiles"]) == len(
            session_info["roles"]
        ), "Number of agent profiles does not match the number of agent roles when initialising agents."
        agent_dict = {}
        for i in range(len(session_info["agent_profiles"])):
            agent_id = session_info["agent_profiles"][i]["ID"]
            agent_role = session_info["roles"][agent_id]
            agent_dict[agent_role] = ConversationalSyntheticAgent(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info=session_info["agent_profiles"][i],
                model_info=self.model_info,
                api_endpoint=self.api_endpoint,
                role=agent_role,
                role_description=self.agent_roles[agent_role],
                treatment=session_info["treatment"],
            )

        return agent_dict

    def _run_session(
        self, session_info: dict[str, Any], test_mode: bool = False
    ) -> dict[str, Any]:
        """Runs a session involving a conversation between multiple AI agents.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only the first session is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        message_history = []
        conversation_length = 0
        num_agents = len(session_info["agents"])
        agent_list = list(session_info["agents"].values())
        response = session_info["session_system_message"]
        agent_role = "system"

        while (
            "Thank you for the conversation" not in response
            and conversation_length < self.max_conversation_length
        ):
            if agent_role == "system" and conversation_length == 0:
                starting_message = [
                    {
                        agent_role: response,
                        "task_id": conversation_length,
                    },
                    {
                        "user": "Start",
                    },
                ]
                message_history.extend(starting_message)
            else:
                message_dict = {
                    agent_role: response,
                    "agent_id": agent.profile_info["ID"],
                    "task_id": conversation_length,
                }
                message_history.append(message_dict)

            if test_mode:
                print(message_dict)
                print()

            # If no interview script is provided, the sequence of conversation will follow the sequence of agents defined in self._initialize_agents
            agent = agent_list[conversation_length % num_agents]

            response = agent.respond(message_history=message_history)
            agent_role = agent.role
            conversation_length += 1

        message_dict = {
            agent_role: response,
            "agent_id": agent.profile_info["ID"],
            "task_id": conversation_length,
        }
        message_history.append(message_dict)
        message_history.append({"system": "End"})
        if test_mode:
            print(message_dict)
            print()
            print({"system": "End"})

        session_info["message_history"] = message_history
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
    specific to AI-to-AI interview experiments. More specifically, one of the agents in the session will serve
    as the facilitator and will not be given a demographic profile.

    Args:
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        agent_roles (dict[str, str]): Dictionary mapping agent roles to their descriptions.
        num_agents_per_session (int, optional): Number of agents per session. Defaults to 2.
        num_sessions (int, optional): Number of sessions. Defaults to 10.
        experiment_context (str, optional): The context or purpose of the experiment. Defaults to an empty string.
        experiment_id (str, optional): The unique ID of the experiment. Defaults to an empty string.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model. Defaults to an empty string.
        max_conversation_length (int, optional): The maximum length of a conversation. Defaults to 10.
        treatments (dict[str, Any], optional): The treatments for the experiment. Defaults to an empty dictionary.
        treatment_assignment_strategy (str, optional): The strategy used for assigning treatments to agents. Defaults to "simple_random".
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments. Defaults to an empty string.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions. Defaults to an empty string.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions. Defaults to "random".
        role_column (str, optional): The column in agent_profiles that contains the manually assigned role. Defaults to an empty string.
        random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        interview_prompts (List[dict[str, str]], optional): An optional dictionary containing the interview script that the facilitator agent has to follow.

    Raises:
        ValueError: If the provided model_info is not supported.
        ValueError: If the provided treatment_assignment_strategy is not supported.
        ValueError: If the provided session_assignment_strategy is not supported.
        ValueError: If the provided role_assignment_strategy is not supported.
        ValueError: If the provided agent_profiles is an empty DataFrame or does not contain a 'ID' column.
        ValueError: If the provided max_conversation_length is lesser than 5.
        ValueError: If the provided treatment is not in the nested dictionary structure when treatment_assignment_strategy is 'full_factorial'.
        ValueError: If the provided num_sessions is not valid.
        ValueError: If the provided num_agents_per_session is less than 2 or will exceed the total number of profile information.
        ValueError: If the provided number of agent_roles is not equal to num_agents_per_session.
        ValueError: If the number of roles defined does not match the number of agents assigned to each session.
        ValueError: If the format of the interview_prompts does not fit with the expected format.

    Attributes:
        model_info (str): The information about the AI model used in the experiment.
        agent_profiles (pd.DataFrame): The profile information of the agents participating in the experiment.
        agent_roles (dict[str, str]): The roles assigned to agents.
        num_agents_per_session (int): The number of agents per session.
        num_sessions (int): The number of sessions in the experiment.
        experiment_context (str): The context or purpose of the experiment.
        experiment_id (str): The unique ID of the experiment.
        api_endpoint (str, optional): The API endpoint for the HuggingFace model.
        max_conversation_length (int): The maximum length of a conversation.
        treatments (dict[str, Any]): The treatments for the experiment.
        treatment_assignment_strategy (str): The strategy used for assigning treatments to agents.
        treatment_column (str, optional): The column in agent_profiles that contains the manually assigned treatments.
        session_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        session_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        role_assignment_strategy (str, optional): The strategy used for assigning agents to sessions.
        role_column (str, optional): The column in agent_profiles that contains the manually assigned sessions.
        random_seed (int, optional): The random seed for reproducibility.
        session_id_list (list): A list of session IDs generated based on the number of sessions.
        treatment_assignment (dict[Any, str]): A dictionary mapping session IDs to treatment labels.
        session_assignment (dict[Any, list[ProfileInfo]]): A dictionary mapping session IDs to a list of agent profile information.
        role_assignment (dict[Any, str]): A dictionary mapping user IDs to a specified role.
        interview_prompts (List[dict[str, str]], optional): An optional dictionary containing the interview script that the facilitator agent has to follow.
    """

    def __init__(
        self,
        model_info: str,
        agent_profiles: pd.DataFrame,
        agent_roles: dict[str, str],
        num_agents_per_session: int = 2,
        num_sessions: int = 10,
        experiment_context: str = "",
        experiment_id: str = "",
        api_endpoint: str = "",
        max_conversation_length: int = 10,
        treatments: dict[str, Any] = {},
        treatment_assignment_strategy: str = "simple_random",
        treatment_column: str = "",
        session_assignment_strategy: str = "random",
        session_column: str = "",
        role_assignment_strategy: str = "random",
        role_column: str = "",
        random_seed: int = 42,
        interview_prompts: List[dict[str, str]] = [],
    ):
        super().__init__(
            model_info,
            agent_profiles,
            agent_roles,
            num_agents_per_session,
            num_sessions,
            experiment_context,
            experiment_id,
            api_endpoint,
            max_conversation_length,
            treatments,
            treatment_assignment_strategy,
            treatment_column,
            session_assignment_strategy,
            session_column,
            role_assignment_strategy,
            role_column,
            random_seed,
        )

        self.num_agents_per_session = self._check_num_agents_per_session(
            num_agents_per_session=num_agents_per_session
        )
        self.agent_roles = self._check_agent_roles(agent_roles=agent_roles)
        self.session_assignment = self._assign_session(random_seed=self.random_seed)
        self.role_assignment = self._assign_role(random_seed=self.random_seed)
        if self.role_assignment_strategy == "manual":
            self._check_manually_assigned_roles()
        self.interview_prompts = self._check_prompts(
            interview_prompts=interview_prompts
        )

    def _check_num_agents_per_session(self, num_agents_per_session: int) -> int:
        """Checks if the provided num_agents_per_session is 2 or more and matches with the number of agent profiles provided.

        Args:
            num_agents_per_session (int): The num_agents_per_session to be checked.

        Returns:
            int: The validated num_agents_per_session.

        Raises:
            ValueError: If the provided num_agents_per_session is not valid.
        """
        if num_agents_per_session < 2:
            raise ValueError(
                f"Invalid num_agents_per_session: {num_agents_per_session}. For AI-AI interview-based experiments, num_agents_per_session should be an integer that is equal to or greater than 2."
            )

        user_defined_roles = [
            role for role in list(self.agent_roles.keys()) if role not in SPECIAL_ROLES
        ]
        if self.num_sessions * len(user_defined_roles) != len(self.agent_profiles):
            raise ValueError(
                f"Total number of user-defined agents required for experiment ({self.num_sessions * len(user_defined_roles)}) does not match with the number of profiles provided in agent_profiles ({len(self.agent_profiles)})."
            )

        return num_agents_per_session

    def _check_agent_roles(self, agent_roles: dict[str, str]) -> dict[str, str]:
        """Checks if the provided agent_roles is valid.

        Args:
            agent_roles (dict[str, str]): The agent_roles to be checked.

        Returns:
            dict[str, str]: The validated agent_roles.

        Raises:
            ValueError: If the provided agent_roles is not valid.
        """
        if len(agent_roles) != self.num_agents_per_session:
            raise ValueError(
                f"Number of roles defined ({len(agent_roles)}) does not match the number of agents assigned to each session ({self.num_agents_per_session})."
            )

        if "Facilitator" not in list(agent_roles.keys()):
            raise ValueError(
                "For an AI-to-AI interview-based experiment, one of the agent roles must be 'Facilitator'."
            )

        return agent_roles

    def _assign_session(self, random_seed: int) -> dict[int, List[ProfileInfo]]:
        """Assigns agent profiles to each session based on the given number of agents per session (minus the special roles) and agent assignment strategy.
        However, if the session_assignment_strategy is 'manual', then assign the agents to their respective sessions based on the
        assignment defined in agent_profiles.

        Args:
            random_seed (int): The random seed for reproducibility.

        Returns:
            dict[int, List[ProfileInfo]]: A dictionary mapping session IDs to a list of agent profile information.
        """
        num_user_defined_roles = len(
            [
                role
                for role in list(self.agent_roles.keys())
                if role not in SPECIAL_ROLES
            ]
        )

        if self.session_assignment_strategy == "manual":
            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_participants = self.agent_profiles[
                    self.agent_profiles[self.session_column] == session_id
                ]

                num_session_participants = len(session_participants)
                if num_session_participants != num_user_defined_roles:
                    raise ValueError(
                        f"Session {session_id} contains {num_session_participants} participants while the number of user-defined roles per session is supposed to be {num_user_defined_roles}"
                    )

                session_assignment[session_id] = session_participants.to_dict(
                    orient="records"
                )

        else:
            randomised_agent_profiles = self.agent_profiles.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

            session_assignment = {}
            for i, session_id in enumerate(self.session_id_list):
                session_assignment[session_id] = randomised_agent_profiles.iloc[
                    i * num_user_defined_roles : (i + 1) * num_user_defined_roles
                ].to_dict(orient="records")

        return session_assignment

    def _assign_role(self, random_seed: int) -> dict[int, str]:
        """Assigns roles to agents based on the specified role assignment strategy.

        Args:
            random_seed (int): The seed value for randomization when using the "random" role assignment strategy.

        Returns:
            dict[int, str]: A dictionary mapping agent IDs to their assigned roles.

        Raises:
            ValueError: If the number of defined roles does not match the number of agents
                        assigned to a session when using the "random" role assignment strategy.
        """
        if self.role_assignment_strategy == "manual":
            role_assignment = self.agent_profiles.set_index("ID")[
                self.role_column
            ].to_dict()

        else:
            random.seed(random_seed)
            role_assignment = {}
            agent_role_labels = [
                role
                for role in list(self.agent_roles.keys())
                if role not in SPECIAL_ROLES
            ]
            for session_id, session_participants in self.session_assignment.items():
                num_participants = len(session_participants)

                if len(agent_role_labels) == num_participants:
                    randomized_roles = random.sample(
                        agent_role_labels, num_participants
                    )

                else:
                    raise ValueError(
                        f"Number of user-defined roles ({len(agent_role_labels)}) does not match the number of agents ({num_participants}) assigned to Session {session_id}."
                    )

                role_assignment.update(
                    {
                        participant["ID"]: role
                        for participant, role in zip(
                            session_participants, randomized_roles
                        )
                    }
                )

        return role_assignment

    def _check_manually_assigned_roles(self) -> None:
        """Validates that all manually assigned roles are defined in the agent roles, excluding special roles like "Facilitator" and "Summariser".

        This method checks whether the roles manually assigned in the `role_assignment`
        dictionary are a subset of the roles defined in the `agent_roles` dictionary.
        If any manually assigned role is not present in the defined agent roles, a
        `ValueError` is raised.

        Raises:
            ValueError: If the roles defined in the `agent_roles` worksheet are not a
            superset of the manually defined roles in the `agent_profiles` worksheet.
        """
        role_label_set = set(
            [
                role
                for role in list(self.agent_roles.keys())
                if role not in SPECIAL_ROLES
            ]
        )
        manual_defined_roles = set(self.role_assignment.values())

        if not role_label_set.issuperset(manual_defined_roles):
            raise ValueError(
                f"The user-defined roles in the agent_roles worksheet ({list[role_label_set]}) is not a superset of the manually defined roles in the agent_profiles worksheet ({list[manual_defined_roles]})."
            )
        else:
            pass

    def _check_prompts(
        self, interview_prompts: List[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Validates and processes a list of interview prompts.

        Args:
            interview_prompts (List[dict[str, Any]]): A list of dictionaries where each dictionary represents a prompt
                with various attributes such as "type", "var_name", "randomized_response_order", etc.
        Returns:
            List[dict[str, Any]]: The validated and processed list of interview prompts.

        Raises:
            ValueError: If any of the following conditions are not met:
                - The `interview_prompts` list is not empty.
                - The first item in `interview_prompts` contains a "type" field with the value "context".
                - The total length of the interview does not exceed the maximum conversation length
                  (calculated as `len(interview_prompts) * self.num_agents_per_session`).
                - Each prompt's "type" field contains only approved prompt types (defined in `SUPPORTED_PROMPT_TYPES`).
                - Each prompt's "var_name" field contains unique variable names.
                - The "randomized_response_order" field contains only approved values (0 or 1).
                - The "validate_response" field contains only approved values (0 or 1).
                - The "generate_speculation_score" field contains only approved values (0 or 1).
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
                'The first item in interview_prompts must contain "type": "context".'
            )

        # Check if the length of the interview would exceed the maximum conversation length
        if (
            len(interview_prompts) * self.num_agents_per_session
            > self.max_conversation_length
        ):
            raise ValueError(
                f"Based on the length of the interview script ({len(interview_prompts)}) and number of agents ({self.num_agents_per_session}), the maximum length of the conversation should be larger or equal to {len(interview_prompts) * self.num_agents_per_session} and not {self.max_conversation_length}."
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

            # Check if the randomized_response_order column contains only approved values (0 or 1)
            if str(prompt_dict["randomized_response_order"]) not in ["0", "1"]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in randomized_response_order field: {prompt_dict['randomized_response_order']}. Supported options include: 0 or 1."
                )
            prompt_dict["randomized_response_order"] = str(
                prompt_dict["randomized_response_order"]
            )

            # Check if the validate_response column contains only approved values (0 or 1)
            if str(prompt_dict["validate_response"]) not in ["0", "1"]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in validate_response field: {prompt_dict['validate_response']}. Supported options include: 0 or 1."
                )
            prompt_dict["validate_response"] = str(prompt_dict["validate_response"])

            # Check if the generate_speculation_score column contains only approved values (0 or 1)
            if str(prompt_dict["generate_speculation_score"]) not in ["0", "1"]:
                raise ValueError(
                    f"Task ID {prompt_dict['task_id']} contains an invalid value in generate_speculation_score field: {prompt_dict['generate_speculation_score']}. Supported options include: 0 or 1."
                )
            prompt_dict["generate_speculation_score"] = str(
                prompt_dict["generate_speculation_score"]
            )

        return interview_prompts

    def run_experiment(
        self,
        test_mode: bool = True,
        version: int = 1,
        save_results_as_csv: bool = False,
    ) -> dict[str, Any]:
        """Runs an experiment based on the experimental settings defined during class initialisation.
        If test_mode is set to True, only the first session will be selected and run; otherwise, sessions are run in parallel.

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
        self.experiment_context = self.interview_prompts.pop(0)["llm_text"]

        # Helper function to process a single session.
        def process_session(session_id: Any) -> tuple[Any, dict]:
            session_info = {}
            session_info["session_id"] = session_id
            session_info["random_seed"] = self.random_seed
            session_info["treatment"] = self.treatments[
                self.treatment_assignment[session_id]
            ]
            session_info["session_system_message"] = (
                generate_conversational_session_system_message(
                    experiment_context=self.experiment_context,
                    treatment=session_info["treatment"],
                )
            )
            session_info["experiment_context"] = self.experiment_context
            session_info["agent_profiles"] = self.session_assignment[session_id]
            session_participant_ids = [
                profile["ID"] for profile in session_info["agent_profiles"]
            ]
            session_info["roles"] = {
                participant_id: assigned_role
                for participant_id, assigned_role in self.role_assignment.items()
                if participant_id in session_participant_ids
            }
            session_info["agents"] = self._initialize_agents(session_info)
            session_info = self._run_session(
                session_info=session_info,
                interview_prompts=self.interview_prompts,
                test_mode=test_mode,
            )
            updated_session_agent = {}
            for agent_role, agent in session_info["agents"].items():
                updated_session_agent[agent_role] = agent.to_dict()
            session_info["agents"] = updated_session_agent

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

    def _initialize_agents(
        self, session_info: dict[str, Any]
    ) -> dict[str, ConversationalSyntheticAgent]:
        """Initializes and returns a dictionary of ConversationalSyntheticAgent objects for both special and user-defined agent roles based on the provided session information.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information, including agent profiles, role, session ID, treatment, etc.

        Returns:
            dict[str, ConversationalSyntheticAgent]: A dictionary where the key indicates the role and the value is an initialized ConversationalSyntheticAgent objects.

        Raises:
            AssertionError: If the number of agent profiles does not match the number of agent roles (minus special roles) when initializing agents.
        """
        agent_dict = {}

        # First, initialise Facilitator agent
        agent_dict["Facilitator"] = ConversationalSyntheticAgent(
            experiment_id=self.experiment_id,
            experiment_context=self.experiment_context,
            session_id=session_info["session_id"],
            profile_info={},
            model_info=self.model_info,
            api_endpoint=self.api_endpoint,
            role=agent_role,
            role_description=self.agent_roles[agent_role],
            treatment=session_info["treatment"],
        )

        # Then, initialise user-defined agents based on sequence defined in agent_profiles
        user_defined_roles = [
            role for role in list(self.agent_roles.keys()) if role not in SPECIAL_ROLES
        ]
        assert len(session_info["agent_profiles"]) == len(
            user_defined_roles
        ), f"Number of agents' profiles ({len(session_info['agent_profiles'])}) does not match the number of user-defined roles ({len(user_defined_roles)}) when initialising agents. The number of agent profiles should be equal the number of user-defined roles (excluding special roles like Facilitator and Summarizer)."

        for i in range(len(session_info["agent_profiles"])):
            agent_id = session_info["agent_profiles"][i]["ID"]
            agent_role = session_info["roles"][agent_id]
            agent_dict[agent_role] = ConversationalSyntheticAgent(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info=session_info["agent_profiles"][i],
                model_info=self.model_info,
                api_endpoint=self.api_endpoint,
                role=agent_role,
                role_description=self.agent_roles[agent_role],
                treatment=session_info["treatment"],
            )

        # Define Summarizer agent last if it is required
        if "Summarizer" in list(self.agent_roles.keys()):
            agent_dict["Summarizer"] = ConversationalSyntheticAgent(
                experiment_id=self.experiment_id,
                experiment_context=self.experiment_context,
                session_id=session_info["session_id"],
                profile_info={},
                model_info=self.model_info,
                api_endpoint=self.api_endpoint,
                role="Summarizer",
                role_description=self.agent_roles["Summarizer"],
                treatment=session_info["treatment"],
            )

        return agent_dict

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
        self, response_options: Any, randomize_response_order: int = False
    ) -> str:
        """Formats the given response options into a human-readable string.

        Args:
            response_options (Any): The response options to format. This can be a range,
                                    a list, or any other type.
            randomize_response_order (int, optional): Indicates whether the response options should be randomized.
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
            return f"Answer with values ranging from {response_options.start} to {response_options.stop - 1}:"

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
                    ", ".join(str(opt) for opt in response_options[:-1])
                    + " or "
                    + str(response_options[-1])
                )
            return f"Answer with {formatted}:"

        # Otherwise, fallback to a simple string conversion.
        else:
            return f"Answer with {str(response_options)}:"

    def _run_session(
        self,
        session_info: dict[str, Any],
        interview_prompts: List[dict[str, str]],
        test_mode: bool = False,
    ) -> dict[str, Any]:
        """Runs a session involving a conversation between multiple AI agents.

        Args:
            session_info (dict[str, Any]): A dictionary containing session information.
            interview_prompts (List[dict[str, str]]): An list containing the interview script that the facilitator agent has to follow.
            test_mode (bool, optional): A boolean indicating if the session is executed under test mode. In test mode, only the first session is executed and all responses are printed out for easy reference.

        Returns:
            dict[str, Any]: A dictionary containing the updated session information at the end of the session.
        """
        message_history = []
        conversation_length = 0
        num_agents = len(session_info["agents"])
        agent_list = list(session_info["agents"].values())
        response = session_info["session_system_message"]
        agent_role = "system"
        message_dict = {agent_role: response}
        message_history.append(message_dict)
        if test_mode:
            print(message_dict)

        if interview_prompts:
            # Sort the order of tasks based on the task_order field. If task order is repeated, then it is expected that the task order are randomised
            interview_prompts = self._sort_tasks(interview_prompts)

            for round in interview_prompts:
                # Facilitator is providing instructions at the beginning to all subjects and allowing the subjects to continue the discussion.
                if round["type"] in ["context", "discussion"]:
                    # Format context/discussion question
                    response = round["llm_text"]["Facilitator"]
                    response = response.replace(
                        "{{response_options}}",
                        self._format_response_options(
                            response_options=round["response_options"]["Facilitator"],
                            randomize_response_order=round["randomized_response_order"],
                        ),
                    )

                    message_dict = {
                        "Facilitator": response,
                        "task_id": round.get("task_id", None),
                        "var_name": round.get("var_name", None),
                    }
                    message_history.append(message_dict)
                    if test_mode:
                        print(message_dict)
                        print()

                    # Loop through each agent and get their response
                    for agent_role, agent in session_info["agents"].items():
                        response = agent.respond(
                            message_history=message_history,
                            generate_speculation_score=round[
                                "generate_speculation_score"
                            ],
                        )

                        message_dict = {
                            agent_role: response,
                            "agent_id": agent.profile_info["ID"],
                            "task_id": round.get("task_id", None),
                            "var_name": round.get("var_name", None),
                        }
                        message_history.append(message_dict)
                        if test_mode:
                            print(message_dict)
                            print()

                elif round["type"] == "question":
                    # Facilitator is posing the same question to each subject.
                    for agent_role, question in session_info["llm_text"].items():
                        # Format interview question
                        question = question.replace(
                            "{{response_options}}",
                            self._format_response_options(
                                response_options=round["response_options"][agent_role],
                                randomize_response_order=round[
                                    "randomized_response_order"
                                ],
                            ),
                        )

                        message_dict = {
                            "Facilitator": question,
                            "task_id": round.get("task_id", None),
                            "var_name": round.get("var_name", None),
                        }
                        message_history.append(message_dict)
                        if test_mode:
                            print(message_dict)
                            print()

                        agent = session_info["agents"][agent_role]
                        response = agent.respond(
                            message_history=message_history,
                            validate_response=round["validate_response"],
                            response_options=round["response_options"][agent_role],
                            generate_speculation_score=round[
                                "generate_speculation_score"
                            ],
                        )

                        message_dict = {
                            agent_role: response,
                            "agent_id": agent.profile_info["ID"],
                            "task_id": round.get("task_id", None),
                            "var_name": round.get("var_name", None),
                        }
                        message_history.append(message_dict)
                        if test_mode:
                            print(message_dict)
                            print()

                else:
                    raise ValueError(
                        f"Invalid prompt type: {round['type']}. The type of prompt for each interview round should one of these options: {SUPPORTED_PROMPT_TYPES}"
                    )

        else:
            while (
                "Thank you for the conversation" not in response
                and conversation_length < self.max_conversation_length
            ):
                message_dict = {
                    agent_role: response,
                    "agent_id": agent.profile_info["ID"],
                    "task_id": conversation_length,
                }
                message_history.append(message_dict)
                if test_mode:
                    print(message_dict)
                    print()
                # If no interview script is provided, the sequence of conversation will follow the sequence of agents defined in self._initialize_agents
                agent = agent_list[conversation_length % num_agents]

                if conversation_length == 0:
                    message_history.append({"user": "Start"})

                response = agent.respond(message_history=message_history)
                agent_role = agent.role
                conversation_length += 1

            message_dict = {
                agent_role: response,
                "agent_id": agent.profile_info["ID"],
                "task_id": conversation_length,
            }
            message_history.append(message_dict)
            if test_mode:
                print(message_dict)
                print()

        message_history.append({"system": "End"})
        if test_mode:
            print({"system": "End"})

        session_info["message_history"] = message_history
        return session_info
