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


    main:
    // put the files in channels
    csv  = Channel.fromPath( input_csv  )
    json = Channel.fromPath( input_json )

    // read the json and create many json as there are combinations of noisers and splitters
    INTERPRET_JSON(json)

    // for each user json many json wiil be created in the same process, but then we need to parallelize and work on them one by one.
    parsed_json = INTERPRET_JSON.out.interpreted_json.flatten()
    
    // launch splitting subworkflow 
    SPLIT_CSV(csv, parsed_json)

    // launch the actual noise subworkflow
    TRANSFORM_CSV( SPLIT_CSV.out.split_data )

    // Launch the shuffle, (always happening on default) and disjointed from split and noise. Data are randomly splitted into this module already.
    // it takes a random json from those interpreted so that is dependant on that process and to have the experiment name key, used later in the train step.
    SHUFFLE_CSV(csv, parsed_json.first())

    // merge output of shuffle to the output of noise
    data = TRANSFORM_CSV.out.transformed_data.concat(SHUFFLE_CSV.out.shuffle_data)

    emit:
    data 

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
