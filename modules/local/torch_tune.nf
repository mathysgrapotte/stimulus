
process TORCH_TUNE {

    tag "${model} - ${combination_key}"
    label 'process_high'
    // TODO: push image to nf-core quay.io
    container "docker.io/mathysgrapotte/stimulus-py:latest"

    input:
    tuple val(combination_key), val(split_transform_key), path(ray_tune_config), path(model), path(data_csv), path(experiment_config), path(initial_weights)

    output:
    tuple val(split_transform_key),
        val(combination_key),
        path(data_csv),
        path(experiment_config),
        path("*-config.json"),
        path("*-model.safetensors"),
        path("*-optimizer.pt"),
        path("*-metrics.csv"),
        path(initial_weights),
        emit: tune_specs

    // output the debug files if they are present, making this an optional channel
    tuple val("debug_${prefix}"),
        path("ray_results/*/debug/best_model_*.txt"),
        path("ray_results/*/debug/worker_with_seed_*/model.safetensors"),
        path("ray_results/*/debug/worker_with_seed_*/seeds.txt"),
        emit: debug,
        optional: true

    script:
    // prefix should be global so that is seen in the output section
    prefix = task.ext.prefix
    def args = task.ext.args ?: ''
    """
    stimulus-tuning \
        -c ${ray_tune_config} \
        -m ${model} \
        -d ${data_csv} \
        -e ${experiment_config} \
        -o ${prefix}-model.safetensors \
        -bo ${prefix}-optimizer.pt \
        -bm ${prefix}-metrics.csv \
        -bc ${prefix}-config.json \
        --initial_weights ${initial_weights} \
        --gpus ${task.accelerator.request} \
        --cpus ${task.cpus} \
        --memory "${task.memory}" \
        $args
    """

    stub:
    def prefix = task.ext.prefix
    """
    touch ${prefix}-model.safetensors ${prefix}-optimizer.pt ${prefix}-metrics.csv ${prefix}-config.json
    """
}
