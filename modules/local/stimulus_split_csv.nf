
process STIMULUS_SPLIT_CSV {
    
    tag "${original_csv} - ${split_transform_key}"
    label 'process_low'
    container 'alessiovignoli3/stimulus:stimulus_v0.3'

    input:
    tuple val(split_transform_key), path(split_json), path(original_csv)

    output:
    tuple val(split_transform_key), path(output), path(split_json), path(original_csv), emit: csv_with_split

    script:
    output = "${original_csv.simpleName}-split.csv"
    """
    launch_split_csv.py -c ${original_csv} -j ${split_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-split.csv"
    """
    touch ${output}
    """
}
