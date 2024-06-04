
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
    # this lione is here just because tune is bugged, if you try to change the location of the ray_results dir without setting tis variable a copy will still be written at the default location home_dir/ray_results 
    export TUNE_RESULT_DIR=\$(pwd)/${suffix}

    launch_check_model.py \
        -d ${original_csv} \
        -m ${model} \
        -e ${experiment_config} \
        -c ${ray_tune_config} \
        $args
    """

    stub:
    """
    echo bubba
    """
}
