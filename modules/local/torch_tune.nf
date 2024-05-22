
process TORCH_TUNE {

    tag "${model}-${combination_key}"
    label 'process_high'
    container "alessiovignoli3/stimulus:stimulus_v0.2"

    input:
    tuple val(combination_key), path(ray_tune_config), path(model), path(data_csv), path(allinfo_json)

    output:
    tuple val(combination_key),
          path(data_csv),
          path(allinfo_json),
          path("*-config.json"),
          path("*-model.pt"),
          path("*-optimizer.pt"),
          path("*-metrics.csv"),
          emit: tune_specs

    script:
    def prefix = task.ext.prefix
    """
    launch_tuning.py \
        -c ${ray_tune_config} \
        -m ${model} \
        -d ${data_csv} \
        -e ${allinfo_json} \
        -o ${prefix}-model.pt \
        -bo ${prefix}-optimizer.pt \
        -bm ${prefix}-metrics.csv \
        -bc ${prefix}-config.json
    """

    stub:
    def prefix = task.ext.prefix
    """
    touch ${prefix}-model.pt ${prefix}-optimizer.pt ${prefix}-metrics.csv ${prefix}-config.json
    """
}
