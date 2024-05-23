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
   
    // 1. Run the default analysis for all models and data together. group them by the same test set -> aka same split process
    ch2default = input_tune_out.groupTuple(by: 1).view()
    /*
    STIMULUS_ANALYSIS_DEFAULT(
        ch2default,
        input_model
    )

    // 2. Run the motif discovery block
    */
}