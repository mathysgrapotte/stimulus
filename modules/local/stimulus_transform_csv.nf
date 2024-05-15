
process STIMULUS_TRANSFORM_CSV {

    tag "$parsed_json"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:stimulus_v0.2'

    input:
    tuple val(original_csv), path(parsed_json), path(splitted_csv)

    output:
    tuple val(original_csv), path(parsed_json), path(output), emit: transformed_data

    script:
    output = "${splitted_csv.baseName}-trans.csv"
    """
    launch_transform_csv.py -c ${splitted_csv} -j ${parsed_json} -o ${output}
    """

    stub:
    output = "${splitted_csv.baseName}-trans.csv"
    """
    touch ${output}
    """
}
