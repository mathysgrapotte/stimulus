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
    INTERPRET_JSON(json, message_from_check)

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
        it -> ["${it.baseName}".split('-')[0..-3].join("-"), "${it.baseName}".split('-')[-1], it]
    }

    //allinfo_json.view()
    split_json.view()
    transform_json.view()
    data = 'bubba'
    /*

    // for each user json many json wiil be created in the same process, but then we need to parallelize and work on them one by one.
    all_exp_json = INTERPRET_JSON.out.interpreted_json.flatten().toSortedList()
    all_exp_json.view()
    // now we need to go from a list to a list of tuples with dir_name, list of files under that dir   as arguments
    grouped_on_split_json = all_exp_json.map {
        it -> ["${it.parent}".split('/')[-1], it]
    }
    //grouped_on_split_json.view()
    // launch splitting subworkflow only once per split configuration/combination. It takes the first file of each tuple.
    parsed_json = grouped_on_split_json.map {
        
        it -> it[1][0]
    }
    //parsed_json.view()
    
    SPLIT_CSV(csv, parsed_json)
    data = SPLIT_CSV.out.split_data
    

    // launch the actual noise subworkflow
    TRANSFORM_CSV( SPLIT_CSV.out.split_data )

    // Launch the shuffle, (always happening on default) and disjointed from split and noise. Data are randomly splitted into this module already.
    // it takes a random json from those interpreted so that is dependant on that process and to have the experiment name key, used later in the train step.
    SHUFFLE_CSV(csv, parsed_json.first())

    // merge output of shuffle to the output of noise
    data = TRANSFORM_CSV.out.transformed_data.concat(SHUFFLE_CSV.out.shuffle_data)
    */

    emit:
    data 

}


/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
