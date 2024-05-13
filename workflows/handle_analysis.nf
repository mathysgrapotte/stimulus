/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { STIMULUS_ANALYSIS_DEFAULT } from '../modules/local/stimulus_analysis_default'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow HANDLE_ANALYSIS {

    take:
    input_tune_out
    input_model

    main:

    // input model channel
    model = Channel.fromPath(input_model)
   
    // 1. Run the default analysis for all models and data together
    input_tune_out
        .groupTuple()
        .set{ ch2default }
    STIMULUS_ANALYSIS_DEFAULT(
        ch2default,
        model
    )

    // 2. Run the motif discovery block

}