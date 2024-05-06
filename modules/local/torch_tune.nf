
process TORCH_TUNE {

    tag "$model-$data_csv"
    label 'process_high'
    container "alessiovignoli3/stimulus:latest"

    input:
    tuple val(original_csv), path(ray_tune_config), path(model), path(data_csv), path(parsed_json)

    output:
    // TODO get the best model as well once implemented in python
    tuple val(original_csv), 
          path("best_config.json"), 
          path("best_model.pt"),
          path("best_optimizer.pt"),
          path("best_metrics.csv"),
          path(data_csv), 
          path(parsed_json), 
          emit: tune_specs

    script:
    """
    launch_tuning.py -c ${ray_tune_config} -m ${model} -d ${data_csv} -j ${parsed_json}
    """

    stub:
    """
    touch best_config.json best_model.pt best_optimizer.pt best_metrics.csv
    """
}