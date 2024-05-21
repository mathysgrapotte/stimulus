
process INTERPRET_JSON {

    tag "$user_json"
    label 'process_low'
    container "alessiovignoli3/stimulus:ps_v0.1"
    
    input:
    path user_json
    val message_from_check_model // only here to ensure that this module waits for check_model module to actually run

    output:
    path ("*-allinfo.json"), emit: allinfo_json
    path ("*-split-*.json"), emit: split_json
    path ("*-transform-*.json"), emit: transform_json

    script:
    """
    launch_interpret_json.py -j ${user_json} 
    """

    stub:
    """
    touch test-#1-allinfo.json
    touch test-#1-split.json
    touch test-#1-transform.json
    touch test-#2-allinfo.json
    touch test-#2-split.json
    touch test-#2-transform.json
    """
}
