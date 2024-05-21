
process STIMULUS_SPLIT_CSV {
    
    tag "$parsed_json"
    label 'process_low'
    container 'alessiovignoli3/stimulus:stimulus_v0.2'

    input:
    tuple val(map_key), path(parsed_json), path(csv)

    output:
    tuple val(map_key), val("${csv}"), path(parsed_json), path(output), emit: csv_with_split

    script:
    output = "${csv.baseName}-${parsed_json.baseName}.csv"
    """
    launch_split_csv.py -c ${csv} -j ${parsed_json} -o ${output}
    """

    stub:
    output = "${csv.baseName}-${parsed_json.baseName}.csv"
    """
    touch ${output}
    """
}
