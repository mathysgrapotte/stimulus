
process INTERPRET_JSON {

    tag "$user_json"
    label 'process_low'
    container "mathysgrapotte/stimulus-py:latest"
    
    input:
    path user_json
    val message_from_check_model // only here to ensure that this module waits for check_model module to actually run

    output:
    path ("*-experiment.json"), emit: experiment_json
    path ("*-split-*.json"), emit: split_json
    path ("*-transform-*.json"), emit: transform_json

    script:
    """
    stimulus-interpret-json -j ${user_json} 
    """

    stub:
    """
    touch test-#1-experiment.json
    touch test-split-null.json
    touch test-#1-transform-null.json
    touch test-#2-experiment.json
    touch test-split-null.json
    touch test-#2-transform-null.json
    """
}
