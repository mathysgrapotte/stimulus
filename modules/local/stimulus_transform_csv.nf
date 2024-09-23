
process STIMULUS_TRANSFORM_CSV {

    tag "${original_csv} - ${combination_key}"
    label 'process_medium'
    container "mathysgrapotte/stimulus-py:latest"

    input:
    tuple val(split_transform_key), val(combination_key), path(transform_json), path(splitted_csv), path(split_json), path(original_csv)

    output:
    // combination_key is put first so that later a combine by:0 can be used to unify with the json that has experiment information (split + transform) associated with this data
    tuple  val(combination_key), val(split_transform_key), path(transform_json), path(output), path(split_json), path(original_csv), emit: transformed_data

    script:
    output = "${original_csv.simpleName}-${combination_key}-trans.csv"
    """
    stimulus-transform-csv -c ${splitted_csv} -j ${transform_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-${combination_key}-trans.csv"
    """
    touch ${output}
    """
}
