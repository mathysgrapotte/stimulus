
process INTERPRET_JSON {

    container "alessiovignoli3/stimulus:ps"
    label 'process_low'

    input:
    path user_json

    output:
    path("json_dir/*.json"), emit: interpreted_json
    stdout emit: standardout

    """
    launch_interpret_json.py -j ${user_json} -d json_dir 
    """

}