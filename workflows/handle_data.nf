/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { INTERPRET_JSON } from '../modules/local/interpret_json.nf'
include { SPLIT_CSV      } from '../subworkflows/split_csv.nf'
include { TRANSFORM_CSV  } from '../subworkflows/transform_csv.nf'
include { SHUFFLE_CSV    } from '../subworkflows/shuffle_csv.nf'


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow HANDLE_DATA {

    take:
    input_csv
    input_json
    message_from_check


    main:
    
    // print the message from the check_model subworkflow 
    message_from_check.view()

    // put the files in channels
    csv  = Channel.fromPath( input_csv  )
    json = Channel.fromPath( input_json )

    // read the json and create many json as there are combinations of noisers and splitters. the message_from_check is passed only to enforce that this modules does not run untill check_module is finished.
    INTERPRET_JSON( json, message_from_check )

    // the above process outputs three channels one with all the information (split+ transform), one with only split info, and one with only transform. Each of this channels have to be transformed into a tuple with a common unique id for a given combination.
    allinfo_json = INTERPRET_JSON.out.allinfo_json.flatten().map{
        it -> ["${it.baseName}".split('-')[0..-2].join("-"), it]
    }

    // the split has only the keyword to match to the transform
    split_json = INTERPRET_JSON.out.split_json.flatten().map{
        it -> ["${it.baseName}".split('-')[-1], it]
    }

    // and transform has both keys to match to everything toghether
    transform_json = INTERPRET_JSON.out.transform_json.flatten().map{
        it -> ["${it.baseName}".split('-')[-1], "${it.baseName}".split('-')[0..-3].join("-"), it]
    }

    // run split with json that only contains experiment name and split information. It runs only the necessary times, all unique ways to split + default split (random split) or column split (already present in data).
    SPLIT_CSV( csv, split_json )

    // assign to each splitted data the associated ransform information based on the split_transform_key generated in the interpret step. 
    transform_split_tuple = transform_json.combine( SPLIT_CSV.out.split_data, by: 0 )

    // launch the actual noise subworkflow
    TRANSFORM_CSV( transform_split_tuple )

    // unify transform output with interpret allinfo json. so that each final data has his own fingerprint json that generated it + keyword. drop all other non relevant fields.
    tmp = allinfo_json.combine( TRANSFORM_CSV.out.transformed_data, by: 0 ).map{
        it -> ["${it[6].name} - ${it[0]}", it[1], it[4]]
    }

    // Launch the shuffle, (always happening on default) and disjointed from split and noise. Data are randomly splitted into this module already.
    // it takes a random json from those interpreted so that is dependant on that process and to have the experiment name key, used later in the train step.
    // It can be still skipped but by default is run. shuffle is set to true in nextflow.config
    data = tmp
    if ( params.shuffle ) {
        SHUFFLE_CSV( csv, allinfo_json.first() )
        // merge output of shuffle to the output of noise
        data = tmp.concat( SHUFFLE_CSV.out.shuffle_data )
    }
    

    emit:
    data 

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
