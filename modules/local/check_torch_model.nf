
process CHECK_TORCH_MODEL {

    tag "$experiment_config - $original_csv"
    label 'process_medium'
    container "alessiovignoli3/stimulus:stimulus_v0.3"
    
    input:
    tuple path(original_csv), path(model),  path(experiment_config), path(ray_tune_config)

    output:
    stdout emit: standardout

    script:
    def suffix = task.ext.suffix
    def args = task.ext.args ?: ''
    """
    #TODO make task.accelerator.request into the variable for gpus number 

    launch_check_model.py \
        -d ${original_csv} \
        -m ${model} \
        -e ${experiment_config} \
        -c ${ray_tune_config} \
        --cpus ${task.cpus} \
        --gpus 0 \
        $args
    """

    stub:
    """
    echo bubba
    """
}
