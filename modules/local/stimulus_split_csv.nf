
process STIMULUS_SPLIT_CSV {
    
    tag "${original_csv} - ${split_transform_key}"
    label 'process_low'
    container "mathysgrapotte/stimulus-py:latest"

    input:
    tuple val(split_transform_key), path(split_json), path(original_csv)

    output:
    tuple val(split_transform_key), path(output), path(split_json), path(original_csv), emit: csv_with_split

    script:
    output = "${original_csv.simpleName}-split.csv"
    """
    stimulus-split-csv -c ${original_csv} -j ${split_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-split.csv"
    """
    touch ${output}
    """
}
