
process CHECK_TORCH_MODEL {

    tag "$experiment_config - $original_csv"
    label 'process_medium'
    container "alessiovignoli3/stimulus:stimulus_v0.3"
    
    input:
    tuple path(original_csv), path(model),  path(experiment_config), path(ray_tune_config)
    each path(initial_weights)

    output:
    stdout emit: standardout

    script:
    def args = task.ext.args ?: ''
    def args_initial_weights = initial_weights ? "--initial_weights $initial_weights" : ""
    """
    launch_check_model.py \
        -d ${original_csv} \
        -m ${model} \
        -e ${experiment_config} \
        -c ${ray_tune_config} \
        ${args_initial_weights} \
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
