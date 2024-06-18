
process STIMULUS_SHUFFLE_CSV {

    tag "${original_csv} - shuffles"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:stimulus_v0.3'

    input:
    tuple val(split_transform_key), path(splitted_csv), path(split_json), path(original_csv)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    tuple val("${original_csv} - shuffle"), val(split_transform_key), path("*.json"), path(output), emit: csv_shuffled

    script:
    output = "${original_csv.simpleName}-shuffle.csv"
    """
    launch_shuffle_csv.py -c ${splitted_csv} -j ${split_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-shuffled.csv"
    """
    touch ${output}
    touch "${original_csv.simpleName}-shuffled-experiment.json"
    """
}
