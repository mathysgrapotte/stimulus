
process STIMULUS_ANALYSIS_DEFAULT {

    tag "${model} - ${split_transform_key}"
    label 'process_medium'
    container "alessiovignoli3/stimulus:stimulus_v0.3"

    input:
    tuple val(split_transform_key), \
          val(combination_key), \
          path(data), \
          path(experiment_config), \
          path(model_config), \
          path(weights), \
          path(optimizer), \
          path(metrics)
    path(model)

    output:
    tuple path("performance_tune_train/"), path("performance_robustness/"), emit: analysis

    script:
    """
    launch_analysis_default.py \
        -m ${model} \
        -w ${weights} \
        -me ${metrics} \
        -ec ${experiment_config} \
        -mc ${model_config} \
        -d ${data} \
        -o .
    """

    stub:
    """
    mkdir performance_tune_train performance_robustness
    """
}
