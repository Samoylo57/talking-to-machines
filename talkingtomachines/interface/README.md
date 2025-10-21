# Prompt Template

## Prompt Template Workbook (`.xlsx`) – Worksheet Overview


| Worksheet Name | Description |
| - | - |
| experimental_setting | Contains the user-defined settings and configurations for the experiment. |
| treatments | Contains the experiment treatment arms and their descriptions. |
| roles | Contains the list of user-defined and special (e.g., Facilitator, Summarizer) role labels and their descriptions. |
| interview_prompts | Contains information about the experiment flow and prompts. |
| demographic_profiles | Contains the synthetic subjects' demographic profile in tabular format. |
| constants | Contains the string/numerical constants that can be dynamically injected into the treatment, agent_roles, interview_prompts worksheets using Jinja. |

*Every sheet name is **case‑sensitive** and **mandatory**. Any additional or missing worksheets will trigger a validation failure.*

---

## 1.  `experimental_setting`

| Key | **Required** | Description/Expected Value |
| - | - | - |
| `experimental_setting` | **Yes** | Serves as the header for the **14 canonical keys** listed below. Expected value: `value`. |
| `experiment_id` | **Yes** | The unique identifier that will be assigned to the experiment. This information will also be used to name the output files after the experiment completes (e.g., <experiment_id>.json and <experiment_id>.csv). |
| `model_info` | **Yes** | The LLM that will be used in the experiment. The platform currently supports most LLMs from OpenAI and Hugging Face. Expected values: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`, `gpt-5-codex`, `gpt-5-pro`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-mini`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`, `hf-inference` (when using the Hugging Face Inference API). Currently, only the LLMs from OpenAI can accept visual inputs. |
| `hf_inference_endpoint` | **Optional** | Refers to the API inference endpoint generated when deploying a Hugging Face Inference Endpoint. This field is only required when choosing `hf-inference` in `model_info`. |
| `temperature` | **Yes** | The temperature setting that will be applied to the LLM. Expected values: Any value between `0-2` (inclusive). This setting is ignored for certain thinking models, including: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-codex`, `gpt-5-pro`, `o1`, `o1-pro`, `o3-pro`, `o3`, `o4-mini`. |
| `num_subjects_per_session` | **Yes** | The number of user-defined subjects that are assigned to each session. This excludes special roles like `Summarizer` and `Facilitator`. |
| `num_sessions` | **Yes** | The number of sessions that will be conducted during the experiment. |
| `max_conversation_length` | **Yes** | The maximum length that the conversation is expected to last for each session. The session will be terminated prematurely if the conversation length exceeds this value. |
| `treatment_assignment_strategy` | **Yes** | The strategy used for assigning treatments to agents. Expected values: `simple_random`, `complete_random`, `manual`. |
| `treatment_column` | **Optional** | In the case that the treatment assignment strategy is `manual`, provide the column name from the `demographic_profiles` worksheet that contains the assigned treatments. |
| `session_assignment_strategy` | **Yes** | The strategy used for assigning subjects to sessions. Expected values: `random`, `manual`. If the treatment assignment strategy is set as `manual`, the session_assignment_strategy must also be set as `manual` to ensure that all subjects in the same session is assigned the same treatment. |
| `session_column` | **Optional** | In the case that the session assignment strategy is `manual`, provide the column name from the `demographic_profiles` worksheet that contains the assigned sessions. |
| `role_assignment_strategy` | **Yes** | The strategy used for assigning roles to subjects. Expected values: `random`, `manual`. |
| `role_column` | **Optional** | In the case that the role assignment strategy is `manual`, provide the column name from the `demographic_profiles` worksheet that contains the assigned roles. |
| `random_seed` | **Optional** | The random seed for reproducibility. Defaults to `42` if not provided. |
| `include_backstories` | **Yes** | A boolean flag for generating backstories based on the subject's demographic information to supplement the subject's profile in the system message. |

---

## 2.  `treatments`

| Column | **Required** | Description |
| - | - | - |
| `treatment_label` | **Yes (Unique)** | A short, concise label for each treatment arm. In the case that the treatment assignment strategy is `manual`, the treatment labels in this worksheet should exactly align with the treatment labels provided in the `demographic_profiles` worksheet.|
| `treatment_description` | **Yes** | A full description of the treatment arm that will be included as part of the LLM-powered subject's system prompt. |

*Extra columns will be rejected. Each row refers to a unique treatment arm.*

---

## 3.  `roles`

| Column | **Required** | Description |
| - | - | - |
| `role_label` | **Yes (Unique)** | A short, concise label for the assigned role. In the case that the role assignment strategy is `manual`, the roles labels in this worksheet should exactly align with the roles labels provided in the `demographic_profiles` worksheet. There are two unique roles that has special abilities (i.e., `Facilitator` and `Summarizer`). The `Facilitator` role controls the flow of the experiment and must be defined in the `demographic_profiles` worksheet for each experiment. The `Summarizer` role can be used to provide summaries at different stages of the experiment or perform intermediate payoff calculations during interactive experiments. The [`public goods experiment demo example`](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment) provides a useful reference on how the `Summarizer` role can be leveraged to calculate participant payoffs at the end of each experiment round. |
| `role_description` | **Yes** | A full description of the role that will be adopted by the LLM. This description will be included as part of the LLM-powered subject's system message. |

*Extra columns will be rejected. Each row refers to a unique agent role.*

---

## 4.  `interview_prompts`

| Column | **Required** | Description/Expected value |
| - | - | - |
| `task_id` | **Yes (Unique)** | A unique identifier for each task. This identifier will be tagged to the LLM's response when generating the output JSON and CSV file if `var_name` is not defined. |
| `type` | **Yes** | The type of tasks that will be conducted during this experiment round. Expected values: `context`, `discussion`, `public_question`, `private_question`. `context` tasks are used to provide the LLM-powered subjects with contextual information about the experiment, and must be defined at the beginning as the first prompt to be incorporated as part of the system prompt. `discussion` tasks are meant to facilitate a group discussion/conversation where a question is posed by the Facilitator to the group at the beginning of the round and the participants will respond sequentially, having visibility of other participants' responses (i.e., Facilitator → Participant 1 → Participant 2 → Participant 3). `public_question` and `private_question` tasks are 1-on-1 type questions that are posed separately to each participant (i.e., Facilitator → Participant 1 → Facilitator → Participant 2 → Facilitator → Participant 3). However, `public_question` tasks are chosen when you want the participants to see their peers' responses within the same round. On the other hand, `private_question` tasks are chosen when you want to hide the participants' responses from other participants. |
| `task_order` | **Yes** | An integer value indicating the order in which the tasks will be executed. If the task order is duplicated across different tasks, then the order of these tasks will be randomized. |
| `is_adapted` | **Yes** | A boolean field indicating if the text from the actual experiment has been adapted. Expected values: `True` or `False`. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `human_text` | **Optional** | The original instructions used in the actual experiment before adaptation. This field is only used for documentation purposes and does not affect the operation of the platform. |
| `llm_text` | **Yes** | The prompt presented to the LLM-powered subjects during each round of the experiment. This could be adapted from the original instructions used in the actual experiment to improve the LLM's performance. The prompt can be defined as a plain string; in that case the same prompt will be automatically presented to each user-defined role listed in the `roles` worksheet. Alternatively, you can define a Python dictionary, where the keys are the role labels (exactly matching those in the `roles` worksheet) and values are the prompt that will be presented to the role. When presenting your prompt as a Python dictionary, you can also customise the role order (based on the order in the dictionary) and also the roles that will participate in this experiment round (i.e., you can exclude certain roles from participating in specific rounds). Additionally, you can pass visual inputs to the LLM-powered subjects by including the URL of a public image in the prompt. Note: Links to images uploaded to Google Drive are currently not supported by the platform as they cannot be accessed by OpenAI's visual models. |
| `var_name` | **Yes (Unique)** | The variable name that will be tagged to the LLM's response when generating the output JSON and CSV files. All variable names should be unique. |
| `var_type` | **Yes** | The expected response type. Expected values: `context`, `category`, `integer`, `free-text`. |
| `response_options` | **Optional** | The response options that will be used to validate the LLM's response during each experiment round. The response options can be defined either as a plain string `Enter a number between 0 and 5`, a Python list `[0,1,2,3,4,5]`, or a Python tuple `(0,5)`. In that case, the same response options will be automatically assigned to every user-defined role listed in the `roles` worksheet. Alternatively, you can define a Python dictionary, where the keys are role labels (exactly matching those in the `roles` worksheet) and values are the response options for that specific role. Similarly, the response options can be a plain string, a Python list, or a Python tuple. When presenting your response options as a Python dictionary, you can customise different action spaces for each user-defined role in that experiment round. |
| `randomize_response_order` | **Yes** | A boolean field indicating if the order of the response options should be randomized before presenting it to the LLM. Expected values: `True` or `False`. |
| `validate_response` | **Yes** | A boolean field indicating if the LLM responses should be validated against the values in the `response_options` field. If the LLM response does not match with any of the options in the `response_options` field, the LLM will be queried again for a maximum of 5 times before proceeding with the last response. Expected values: `True` or `False`. |
| `generate_speculation_score` | **Yes** | A boolean field indicating if the LLM should generate a speculation score (where 0 = not speculative at all and 100 = entirely speculative.). This is used to guard against LLM hallucination. Expected values: `True` or `False`. |
| `format_response` | **Yes** | A boolean field indicating if the LLM response should be formatted as a JSON string or plain text string. Expected values: `True` or `False`. |

*Extra columns will be rejected. Each row refers to a new task/round in the experiment.*

---

## 5.  `demographic_profiles`

* **Row 1:** Shorten name for the demographic question. *Must be non‑blank & unique.*
* **Row 2:** The actual wording used in the demographic question. *Must be non‑blank and human-readable.*
* **Row 3 … n:** The subjects' demographic data, where each row represent the profile of a unique subject and each column refers to the response provided by the subject for a particular question.
* There must be a column named 'ID' representing a unique identifier for each subject's profile.
* Each subject's responses are merged into a single Q&A text snippet—`Interviewer: {question}` followed by `Me: {response}`—that records their demographic profile. This snippet is then inserted into the LLM-powered subject's system message to provide the LLM context about the subject's demographics. If `include_backstories` is set to `True`, the subject's backstory will be generated based on the subject's responses recorded in this worksheet.


---

## 6.  `constants`

| Column | **Required** | Description |
| - | - | - |
| `name` | **Yes (Unique)** | The template that will be used by Jinja to identify and replace the placeholders in the `treatment`, `roles`, `interview_prompts` worksheets (e.g., {{placeholder}}). |
| `value` | **Yes** | Expects a list containing different permutations that should be applied to the placeholders. |

*Extra columns will be rejected. Each row refers to a new constant permutation. If more than one row is defined, the package will perform a cartesian product over all rows to create a list of all possible permutations. Each permutation will spin off a separate experiment.*

---

## Demo Examples
* **Public Goods Experiment**: A public goods experiment demo example with a populated prompt template workbook and description of its experimental design: [Public Goods Experiment Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/public_good_experiment)

* **Randomized Controlled Trial (RCT)**: A RCT experiment demo example with a populated prompt template workbook: [RCT Demo](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/rct_experiment)

* **Prompt Template**: Alternatively, a blank version of the prompt template has been provided to serve as a starting point for creating new synthetic experiments: [Prompt Template](https://github.com/talking-to-machines/talking-to-machines/tree/main/demos/prompt_template.xlsx)

---

## Video Walkthrough
A video walkthrough on how to populate the prompt template workbook based on a simple public goods experiment can be found here: [Video Walkthrough](https://www.loom.com/share/ba5c913979344fd384fd769c64c01cf4?sid=7e2a8981-4826-4230-858d-e1fd63894157)