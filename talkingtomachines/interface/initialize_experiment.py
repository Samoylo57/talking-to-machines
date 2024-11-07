from talkingtomachines.management.experiment import AItoAIInterviewExperiment


def initialize_experiment(prompt_template_data: dict) -> dict:
    experiment = None
    # experiment = AItoAIInterviewExperiment(
    #     experiment_id="reducing_political_polarisation_demo_placebo_immigration_pretreatment",
    #     model_info=model_info,
    #     experiment_context=experiment_context,
    #     agent_demographics=placebo_immigration_data[demographic_cols + ["chatID_full"]],
    #     agent_roles=agent_roles,
    #     num_agents_per_session=2,
    #     num_sessions=len(placebo_immigration_session_ids),
    #     treatments=treatments,
    #     agent_assignment_strategy="manual",
    #     session_column="chatID_full",
    #     interview_script=placebo_immigration_pretreatment_interview_script
    # )

    return experiment
