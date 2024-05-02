
process STIMULUS_SHUFFLE_CSV {

    container 'alessiovignoli3/stimulus:latest'
    label 'process_medium'

    input:
    tuple path(original_csv), path(random_parsed_json)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    tuple val("${original_csv}"), path("*.json"), path(output), emit: csv_shuffled
    stdout emit: standardout

    script:
    output = "${original_csv.baseName}-shuffled.csv"
    """
    launch_shuffle_csv.py -c ${original_csv} -j ${random_parsed_json} -o ${output}
    """
}