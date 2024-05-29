
process TORCH_TUNE {

    tag "${model} - ${combination_key}"
    label 'process_high'
    container "alessiovignoli3/stimulus:stimulus_v0.2"

    input:
    tuple val(combination_key), val(split_transform_key), path(ray_tune_config), path(model), path(data_csv), path(experiment_config)

    output:
    tuple val(split_transform_key),
          val(combination_key),
          path(data_csv),
          path(experiment_config),
          path("*-config.json"),
          path("*-model.pt"),
          path("*-optimizer.pt"),
          path("*-metrics.csv"),
          emit: tune_specs

    script:
    def prefix = task.ext.prefix
    def suffix = task.ext.suffix
    def args = task.ext.args ?: ''
    """
    # this lione is here just because tune is bugged, if you try to change the location of the ray_results dir without setting tis variable a copy will still be written at the default location home_dir/ray_results 
    export TUNE_RESULT_DIR=\$(pwd)/${suffix}
    
    launch_tuning.py \
        -c ${ray_tune_config} \
        -m ${model} \
        -d ${data_csv} \
        -e ${experiment_config} \
        -o ${prefix}-model.pt \
        -bo ${prefix}-optimizer.pt \
        -bm ${prefix}-metrics.csv \
        -bc ${prefix}-config.json \
        $args
    """

    stub:
    def prefix = task.ext.prefix
    """
    touch ${prefix}-model.pt ${prefix}-optimizer.pt ${prefix}-metrics.csv ${prefix}-config.json
    """
}
