
process STIMULUS_SPLIT_CSV {

    container 'alessiovignoli3/stimulus:latest'
    label 'process_low'

    input:
    tuple path(csv), path(parsed_json)

    output:
    tuple val("${csv}"), path(parsed_json), path(output), emit: csv_with_split
    stdout emit: standardout

    script:
    output = "${csv.baseName}-${parsed_json.baseName}.csv"
    """
    launch_split_csv.py -c ${csv} -j ${parsed_json} -o ${output}
    """
}
