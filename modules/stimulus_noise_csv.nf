
process STIMULUS_NOISE_CSV {

    container 'alessiovignoli3/stimulus:stimulus_noise'
    label 'process_medium'

    input:
    tuple val(original_csv), path(parsed_json), path(splitted_csv)

    output:
    tuple val(original_csv), path(parsed_json), path(output), emit: noised_data
    stdout emit: standardout

    script:
    output = "${splitted_csv.baseName}-noised.csv"
    """
    launch_noise_csv.py -c ${splitted_csv} -j ${parsed_json} -o ${output}
    """
}
