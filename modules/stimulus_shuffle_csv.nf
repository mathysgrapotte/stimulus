
process STIMULUS_SHUFFLE_CSV {

    container 'alessiovignoli3/stimulus:stimulus_noise'

    input:
    tuple val(original_csv), path(random_parsed_json)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    //tuple val(original_csv), path(artificial_json), path(output), emit: shuffled_data
    stdout emit: standardout

    script:
    output = "${original_csv.baseName}-shuffled.csv"
    """
    launch_shuffle_csv.py -c ${original_csv} -j ${random_parsed_json} -o ${output}
    """
}