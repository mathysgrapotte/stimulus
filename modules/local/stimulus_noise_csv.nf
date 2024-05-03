
process STIMULUS_NOISE_CSV {

    tag "$parsed_json"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:latest'

    input:
    tuple val(original_csv), path(parsed_json), path(splitted_csv)

    output:
    tuple val(original_csv), path(parsed_json), path(output), emit: noised_data

    script:
    output = "${splitted_csv.baseName}-noised.csv"
    """
    launch_data_transformer.py -c ${splitted_csv} -j ${parsed_json} -o ${output}
    """

    stub:
    output = "${splitted_csv.baseName}-noised.csv"
    """
    touch ${output}
    """
}
