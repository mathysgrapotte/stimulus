
process STIMULUS_SHUFFLE_CSV {

    tag "${original_csv} - shuffles"
    label 'process_medium'
    // TODO: push image to nf-core quay.io
    container "docker.io/mathysgrapotte/stimulus-py:latest"

    input:
    tuple val(split_transform_key), path(splitted_csv), path(split_json), path(original_csv)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    tuple val("${original_csv} - shuffle"), val(split_transform_key), path("*.json"), path(output), emit: csv_shuffled

    script:
    output = "${original_csv.simpleName}-shuffle.csv"
    """
    stimulus-shuffle-csv -c ${splitted_csv} -j ${split_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-shuffled.csv"
    """
    touch ${output}
    touch "${original_csv.simpleName}-shuffled-experiment.json"
    """
}
