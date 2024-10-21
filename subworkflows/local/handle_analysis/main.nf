/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { STIMULUS_ANALYSIS_DEFAULT } from '../../../modules/local/stimulus_analysis_default'

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

    // 1. Run the default analysis for all models and data together. group them by the same test set -> aka same split process
    input_tune_out
        .map{it -> [it[0], it[1], it[2], it[3], it[4], it[5], it[6], it[7]]}   // cannot work with empty initial_weights now. TODO fix this in the future
        .groupTuple()
        .set{ ch2default }
    ch2default.view()

    STIMULUS_ANALYSIS_DEFAULT(
        ch2default,
        input_model
    )

    // 2. Run the motif discovery block

}
