# Prompt Template

## Prompt Template Workbook (`.xlsx`) – Worksheet Overview

```
experimental_setting   ← contains the user-defined settings for the experiment.
treatments             ← contains the experiment treatment arms.
agent_roles            ← contains the list of agent role labels and their descriptions.
interview_prompts      ← contains information about the experiment flow and prompts.
agent_profiles         ← contains the synthetic agents' metadata in tabular format.
constants              ← contains the string/numerical constants that can be dynamically injected into the treatment, agent_roles, interview_prompts worksheets using Jinja.
```

*Every sheet name is **case‑sensitive** and **mandatory**. Any additional or missing worksheets will trigger a validation failure.*

---

## 1.  `experimental_setting`

|Key|**Required**|Description/Expected value|
| - | - | - |
|`experimental_setting`|**Yes**| Serves as the header for the **14 canonical keys** listed below. Expected value: 'value'.|
|`experiment_id`|**Yes**|The unique identifier that will be assigned to the experiment. This will also be used when naming the output files after the experiment completes (e.g., <experiment_id>.json and <experiment_id>.csv).|
|`model_info`|**Yes**|The LLM that will be used in the experiment. Currently, only the LLMs from OpenAI and Hugging Face is supported. Expected values: 'gpt-4.5-preview', 'o3', 'o4-mini', 'o1-pro', 'o1', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o' 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 'hf-inference' (when using Hugging Face inference API). If the model chosen falls outside of this list, a warning will be reflected and defaults to using the OpenAI API. |
|`api_endpoint`|**Optional**|Refers to the API endpoint generated when deploying a Hugging Face Inference Endpoint. This field is only required when choosing 'hf-inference' in `model_info`.|
|`temperature`|**Yes**|The temperature setting that will be applied to the LLM. Expected values: Any value between 0-2 (inclusive).|
|`num_agents_per_session`|**Yes**|The number of user-defined agents that is assigned to each session. Excludes special agent roles like 'Summarizer' and 'Facilitator'.|
|`num_sessions`|**Yes**|The number of sessions that will be conducted during the experiment.|
|`max_conversation_length`|**Yes**|The maximum length that the conversation is expected to last for each session.|
|`treatment_assignment_strategy`|**Yes**|The strategy used for assigning treatments to agents. Expected values: 'simple_random', 'complete_random', 'manual'.|
|`treatment_column`|**Optional**|In the case that the treatment assignment strategy is 'manual', provide the column name from the 'agent_profiles' worksheet that contains the assigned treatments.|
|`session_assignment_strategy`|**Yes**|The strategy used for assigning agents to sessions. Expected values: 'random', 'manual'.|
|`session_column`|**Optional**|In the case that the session assignment strategy is 'manual', provide the column name from the 'agent_profiles' worksheet that contains the assigned sessions.|
|`role_assignment_strategy`|**Yes**|The strategy used for assigning roles to agents. Expected values: 'random', 'manual'.|
|`role_column`|**Optional**|In the case that the role assignment strategy is 'manual', provide the column name from the 'agent_profiles' worksheet that contains the assigned roles.|
|`random_seed`|**Optional**|The random seed for reproducibility. Defaults to 42 if not provided.|

---

## 2.  `treatments`

|Column|**Required**|Description|
|-|-|-|
|`treatment_label`| **Yes (unique)**| A short, concise label for the treatment arm. In the case that the 'treatment_assignment_strategy' is manual, the treatment labels in this worksheet should exactly align with the treatment labels provided in the 'agent_profiles' worksheet.|
|`treatment_description`|**Yes**| A full description of the treatment arm that will be presented to the LLM.|

*Extra columns will be rejected. Each row refers to a unique treatment arm.*

---

## 3.  `agent_roles`

|Column|**Required**|Description|
|-|-|-|
|`role_label`| **Yes (unique)**| A short, concise label for the assigned role. In the case that the 'role_assignment_strategy' is manual, the roles labels in this worksheet should exactly align with the roles labels provided in the 'agent_profiles' worksheet. There are two unique roles that has special abilities (e.g., Facilitator and Summarizer). The 'Facilitator' role controls the flow of the experiment and must be defined for each experiment. The 'Summarizer' role can be used to provide summaries at different stages of the experiment or perform intermediate payoff calculations during interactive experiments.|
|`role_description`|**Yes**| A full description of the role that will be adopted by the LLM.|

*Extra columns will be rejected. Each row refers to a unique agent role.*

---

## 4.  `interview_prompts`

|Column|**Required**|Description/Expected value|
|-|-|-|
|`task_id`|**Yes (unique)**|A unique identifier for each task. This identifier will be tagged to the LLM's response when generating the output JSON and CSV file if `var_name` is not defined.|
|`type`|**Yes**|The type of task that will be conducted during this experiment round. Expected values: 'context', 'discussion', 'public_question', 'private_question'. **context** tasks are used to provide the LLM with contextual information about the experiment, and must be defined at the beginning as the first task to be incorporated in the LLM's system message. **discussion** tasks are meant to facilitate a group discussion/conversation where a question is posed to the group at the beginning of the round and the participants will respond sequentially (i.e., Facilitator → Participant 1 → Participant 2 → Participant 3). **public_question** and **private_question** tasks are one-on-one type questions that will be posed separately to each participant (i.e., Facilitator → Participant 1 → Facilitator → Participant 2 → Facilitator → Participant 3). However, **public_question** tasks are chosen when you want the participants to see their peers' answers within the same round. On the other hand, **private_question** tasks are chosen when you want to hide the participants' responses from others until the round ends.|
|`task_order`|**Yes**|A integer value indicating the order in which the tasks will be executed. If the order value is duplicated across different tasks, then the order of these tasks will be randomized.|
|`is_adapted`|**Yes**|An boolean field indicating if the text from the actual experiment has been adapted. Expected values: '0', '1'.|
|`human_text`|**Optional**|The original wording used in the actual experiment.|
|`llm_text`|**Yes**|The prompt presented to the LLM during each round of the experiment. The prompt can be defined as a plain string; in that case the same prompt will be automatically presented to each user-defined role listed in the 'agent_roles' worksheet. Alternatively, you can define a Python dictionary, where the keys are role labels (exactly matching those in 'agent-roles') and values are the prompt text presented to that role. When presenting your prompt as a Python dictionary, you can customise the role order (based on the dictionary's order) and also the roles that will participate in this round.|
|`var_name`|**Yes (unique)**|The variable name that will be tagged to the LLM's response when generating the output JSON and CSV files. All variable names should be unique.|
|`var_type`|**Optional**|The expected response type. Expected values: 'category', 'integer', 'float'.|
|`response_options`|**Optional**|The response options presented to the LLM during each round of the experiment for validation. The response options can be defined as either a plain string (e.g., "Enter a number between 0 and 5"), a Python list (e.g., [0,1,2,3,4,5]), or a Python tuple (e.g., (0,5)). In that case, the same response options will be automatically assigned to every user-defined role listed in the 'agent_roles' worksheet. Alternatively, you can define a Python dictionary, where the keys are role labels (exactly matching those in the 'agent_roles' worksheet) and values are the response options for that role. Similarly, the response options can be a plain string, a Python list, or a Python tuple. When presenting your response options as a Python dictionary, you can customise different action spaces for each role in that round.|
|`randomize_response_order`|**Yes**| A boolean field indicating if the order of the response options should be randomized before presenting it to the LLM. Expected values: '0', '1'.|
|`validate_response`|**Yes**|A boolean field indicating if the LLM responses should be validated against the values in the `response_options` field. If the LLM response does not match with any of the options in the `response_options` field, the LLM will be queried again for a maximum of 5 times before proceeding with the last response. Expected values: '0', '1'.|
|`generate_speculation_score`|**Yes**|A boolean field indicating if the LLM should generate a speculation score (where 0 = not speculative at all and 100 = entirely speculative.). This is used to guard against LLM hallucination. Expected values: '0', '1'.|
|`format_response`|**Yes**|A boolean field indicating if the LLM response should be formatted as a JSON string or plain text string. Expected values: '0', '1'.|

*Extra columns will be rejected. Each row refers to a new task/round in the experiment.*

---

## 5.  `agent_profiles`

* **Row 1:** Shorten name for the actual survey question. *Must be non‑blank & unique.*
* **Row 2:** The actual survey question. *Must be non‑blank and human-readable.*
* **Row 3 … n:** Actual agent data, where each row represent the profile of a unique participant and each column refers to the response provided by the participant for a particular survey question.
* There must be a column named 'ID' representing a unique identifier for each participant profile.

---

## 6.  `constants`

|Column|**Required**|Description|
|-|-|-|
|`name`|**Yes (unique)**|The template that will be used by Jinja to identify and replace the placeholders in the 'treatment', 'agent_roles', 'interview_prompts' worksheets (e.g., {{placeholder}}).|
|`value`|**Yes**|Expects a list containing the different permutations that should be applied to the placeholders.|

*Extra columns will be rejected. Each row refers to a new constant permutation. If more than one row is defined, the package will perform a cartesian product over all rows to create a list of all possible permutations. Each permutation will spin off a new experiment.*

---

## Demo Example
A complete public goods experiment demo example—containing a populated prompt template and description of its experimental design has been provided for your reference:
https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment

Alternatively, a blank version of the prompt template has also been provided to serve as a starting point for creating new templates:
https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/prompt_template.xlsx
