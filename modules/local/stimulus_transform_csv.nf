
process STIMULUS_TRANSFORM_CSV {

    tag "${original_csv} - ${combination_key}"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:stimulus_v0.2'

    input:
    tuple val(split_transform_key), val(combination_key), path(transform_json), path(splitted_csv), path(split_json), path(original_csv)

    output:
    // combination_key is put fist so that later a combine by:0 can be used to unify with the json that has allinformation (split + transform) associated with this data
    tuple  val(combination_key), val(split_transform_key), path(transform_json), path(output), path(split_json), path(original_csv), emit: transformed_data

    script:
    output = "${original_csv.simpleName}-trans.csv"
    """
    launch_transform_csv.py -c ${splitted_csv} -j ${transform_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-trans.csv"
    """
    touch ${output}
    """
}
