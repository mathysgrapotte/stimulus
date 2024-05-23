
process STIMULUS_SHUFFLE_CSV {

    tag "${original_csv}"
    label 'process_medium'
    container 'alessiovignoli3/stimulus:stimulus_v0.2'

    input:
    tuple val(random_combination_key), path(random_parsed_json), path(original_csv)

    output:
    // this type of output is so it is more easily unifiable with the output of the noise module.
    tuple val("${original_csv} - shuffle"), path("*.json"), path(output), emit: csv_shuffled

    script:
    output = "${original_csv.simpleName}-shuffle.csv"
    """
    launch_shuffle_csv.py -c ${original_csv} -j ${random_parsed_json} -o ${output}
    """

    stub:
    output = "${original_csv.simpleName}-shuffled.csv"
    """
    touch ${output}
    touch "${original_csv.simpleName}-shuffled-allinfo.json"
    """
}
