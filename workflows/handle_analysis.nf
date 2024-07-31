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
    input_initial_weights

    main:

   
    // 1. Run the default analysis for all models and data together. 
    // they are already grouped by the same test set -> aka same split process. Only models that have exactly the same test set can be compared among themselves. Comparison of model preformances on different sets is biased.
    // add the initial_weights if present
    if (input_initial_weights == null){
        // the empty list will be passed to path operator which will return "" and simply not pass anything to the louncher
        input_tune_out = input_tune_out.map{ it -> [ it[0], it[1], it[2], it[3], it[4], it[5], it[6], it[7], [] ] }
    }else{
        // the initial_weights files are all pased as a list to each analysis task. TODO see if they are needed or not and how usable they are.
        initial_weights = Channel.fromPath( input_initial_weights ).toList().map{ it -> [ it ]}
        input_tune_out = input_tune_out.combine(initial_weights)
    }

    input_tune_out.view()

    
    STIMULUS_ANALYSIS_DEFAULT(
        input_tune_out,
        input_model
    )
    
    // 2. Run the motif discovery block
    
}