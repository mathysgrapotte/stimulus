process STIMULUS_SPLIT_CSV {
    
    tag "$csv"
    label 'process_low'
    container 'alessiovignoli3/stimulus:latest'

    input:
    tuple path(csv), path(parsed_json)

    output:
    tuple val("${csv}"), path(parsed_json), path(output), emit: csv_with_split

    script:
    output = "${csv.baseName}-${parsed_json.baseName}.csv"
    """
    launch_split_csv.py -c ${csv} -j ${parsed_json} -o ${output}
    """

    stub:
    output = "${csv.baseName}-${parsed_json.baseName}.csv"
    """
    touch ${output}
    echo "launch_split_csv.py -c ${csv} -j ${parsed_json} -o ${output}" > launch_split_csv.sh
    """
}
