import itertools
from talkingtomachines.management.experiment import AItoAIInterviewExperiment
from jinja2 import Template


def render_dict_with_template(template_dict: dict, constants_dict: dict) -> dict:
    """Renders a dictionary with template strings using the provided constants.

    This function takes a dictionary where some values are template strings
    and another dictionary with constants to replace in the template strings.
    It processes the dictionary recursively, rendering all template strings
    with the provided constants.

    Args:
        template_dict (dict): The dictionary containing template strings.
        constants_dict (dict): The dictionary containing constants to replace
                               in the template strings.

    Returns:
        dict: A new dictionary with all template strings rendered with the
              provided constants.
    """
    rendered_dict = {}
    for key, value in template_dict.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            rendered_dict[key] = render_dict_with_template(value, constants_dict)

        elif isinstance(value, str):
            # Render template for each string value
            template = Template(value)
            rendered_value = template.render(constants_dict)
            rendered_dict[key] = rendered_value

        elif isinstance(value, list):
            # Render template for each item in the list
            rendered_list = []
            for item in value:
                if isinstance(item, dict):
                    # Recursively process nested dictionaries
                    rendered_list.append(
                        render_dict_with_template(item, constants_dict)
                    )

                elif isinstance(item, str):
                    # Render template for each string value
                    template = Template(item)
                    rendered_value = template.render(constants_dict)
                    rendered_list.append(rendered_value)

                else:
                    rendered_list.append(item)

            rendered_dict[key] = rendered_list

        else:
            rendered_dict[key] = value

    return rendered_dict


def generate_permutations(constants: dict) -> list:
    """Generate all possible permutations of the given list of constants.

    This function takes a dictionary of constants where the keys are the names of the constants
    and the values are lists of possible values for those constants. It returns a list of dictionaries,
    each representing a unique permutation of the constants.

    Args:
        constants (dict): A dictionary where keys are constant names and values are lists of possible values.

    Returns:
        list: A list of dictionaries, each containing a unique permutation of the constants.
    """
    if constants:
        keys, values = zip(*constants.items())
        constant_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return constant_permutations
    else:
        return [{}]


def initialize_experiment(prompt_template_data: dict) -> list:
    """Initializes a list of AI-to-AI interview experiments based on the provided prompt template data.

    Args:
        prompt_template_data (dict): A dictionary containing the prompt template data.
                                     It should include a "constants" key with values to be permuted.

    Returns:
        list: A list of initialized AItoAIInterviewExperiment objects.
    """
    # Define all constant permutations
    constant_permutations = generate_permutations(prompt_template_data["constants"])

    experiments = []
    for constant_permutation in constant_permutations:
        # For each permutation, apply constants to prompt template using Jjanja
        rendered_prompt_template = render_dict_with_template(
            prompt_template_data, constant_permutation
        )

        # Initialise experiment based on rendered prompt template
        experiment = AItoAIInterviewExperiment(
            experiment_id=rendered_prompt_template["experiment_id"],
            model_info=rendered_prompt_template["model_info"],
            api_endpoint=rendered_prompt_template["api_endpoint"],
            agent_profiles=rendered_prompt_template["agent_profiles"],
            agent_roles=rendered_prompt_template["agent_roles"],
            num_agents_per_session=rendered_prompt_template["num_agents_per_session"],
            num_sessions=rendered_prompt_template["num_sessions"],
            max_conversation_length=rendered_prompt_template["max_conversation_length"],
            treatments=rendered_prompt_template["treatments"],
            treatment_assignment_strategy=rendered_prompt_template[
                "treatment_assignment_strategy"
            ],
            agent_assignment_strategy=rendered_prompt_template[
                "agent_assignment_strategy"
            ],
            treatment_column=rendered_prompt_template["treatment_column"],
            session_column=rendered_prompt_template["session_column"],
            experiment_prompts=rendered_prompt_template["experiment_prompts"],
        )

        experiments.append(experiment)

    return experiments
