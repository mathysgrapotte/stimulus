
process INTERPRET_JSON {

    tag "$user_json"
    label 'process_low'
    container "alessiovignoli3/stimulus:ps"
    
    input:
    path user_json

    output:
    path("json_dir/*.json"), emit: interpreted_json

    script:
    """
    launch_interpret_json.py -j ${user_json} -d json_dir 
    """

    stub:
    """
    mkdir json_dir
    touch json_dir/test-#1.json
    touch json_dir/test-#2.json
    """
}
