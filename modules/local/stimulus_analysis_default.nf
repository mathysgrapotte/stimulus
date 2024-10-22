
process STIMULUS_ANALYSIS_DEFAULT {

    tag "${model} - ${split_transform_key}"
    label 'process_medium'
    // TODO: push image to nf-core quay.io
    container "docker.io/mathysgrapotte/stimulus-py:latest"

    input:
    tuple val(split_transform_key), val(combination_key), path(data), path(experiment_config), path(model_config), path(weights), path(optimizer), path(metrics)
    path(model)

    output:
    tuple path("performance_tune_train/"), path("performance_robustness/"), emit: analysis

    script:
    """
    stimulus-analysis-default \
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
