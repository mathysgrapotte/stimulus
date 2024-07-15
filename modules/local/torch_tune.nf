
process TORCH_TUNE {

    tag "${model} - ${combination_key}"
    label 'process_high'
    container "alessiovignoli3/stimulus:stimulus_v0.3"

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
    // output the debug files if they are present
     

    script:
    def prefix = task.ext.prefix
    def suffix = task.ext.suffix
    def args = task.ext.args ?: ''
    """
    launch_tuning.py \
        -c ${ray_tune_config} \
        -m ${model} \
        -d ${data_csv} \
        -e ${experiment_config} \
        -o ${prefix}-model.pt \
        -bo ${prefix}-optimizer.pt \
        -bm ${prefix}-metrics.csv \
        -bc ${prefix}-config.json \
        --gpus ${task.accelerator.request} \
        --cpus ${task.cpus} \
        --memory "${task.memory}" \
        $args
    """

    stub:
    def prefix = task.ext.prefix
    """
    touch ${prefix}-model.pt ${prefix}-optimizer.pt ${prefix}-metrics.csv ${prefix}-config.json
    """
}
