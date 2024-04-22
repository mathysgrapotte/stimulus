
process TORCH_TRAIN {

    container "alessiovignoli3/stimulus:torch_train"

    input:
    tuple val(original_csv), path(ray_tune_config), path(model), path(data_csv), path(parsed_json)

    output:
    stdout emit: standardout

    script:
    //output = "${data_csv.baseName}-${parsed_json.baseName}.csv"
    """
    launch_training.py -c ${ray_tune_config} -m ${model} -d ${data_csv} -j ${parsed_json}
    """
}