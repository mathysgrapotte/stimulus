
process STIMULUS_SHUFFLE_CSV {

    tag "$random_parsed_json"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:stimulus_v0.2'

    input:
    tuple path(original_csv), path(random_parsed_json)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    tuple val("${original_csv}"), path("*.json"), path(output), emit: csv_shuffled

    script:
    output = "${original_csv.baseName}-shuffled.csv"
    """
    launch_shuffle_csv.py -c ${original_csv} -j ${random_parsed_json} -o ${output}
    """

    stub:
    output = "${original_csv.baseName}-shuffled.csv"
    """
    touch ${output}
    touch "${original_csv.baseName}-shuffled.json"
    """
}
