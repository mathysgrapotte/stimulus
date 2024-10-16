
process CHECK_TORCH_MODEL {

    tag "$experiment_config - $original_csv"
    label 'process_medium'
    container "mathysgrapotte/stimulus-py:latest"
    
    input:
    tuple path(original_csv), path(model),  path(experiment_config), path(ray_tune_config), path(initial_weights)

    output:
    stdout emit: standardout

    script:
    def args = task.ext.args ?: ''
    """
    stimulus-check-model \
        -d ${original_csv} \
        -m ${model} \
        -e ${experiment_config} \
        -c ${ray_tune_config} \
        --initial_weights ${initial_weights} \
        --gpus ${task.accelerator.request} \
        --cpus ${task.cpus} \
        --memory "${task.memory}" \
        $args
    """

    stub:
    """
    echo bubba
    """
}
