/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { STIMULUS_SHUFFLE_CSV } from '../modules/stimulus_shuffle_csv.nf'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN SUBWORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow SHUFFLE_CSV {

    take:
    data_csv
    json
    

    main:

    // if there is more than one csv then each of them will be associated to all Json. This means all the modifications will be made on all the input csv.
    csv_json_pairs = data_csv.combine(json)
    
    // It can be still skipped but by default is run. shuffle is set to true in nextflow.config
    if ( params.shuffle ) {
        STIMULUS_SHUFFLE_CSV(csv_json_pairs)
    }


    emit:

    debug         = STIMULUS_SHUFFLE_CSV.out.standardout
    shuffle_data  = STIMULUS_SHUFFLE_CSV.out.csv_shuffled

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
