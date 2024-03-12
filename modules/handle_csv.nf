
process HANDLE_CSV {

    container 'alessiovignoli3/stimulus:torch_scikit_numpy'

    input:
    path csv
    path json

    output:
    stdout emit: standardout

    script:
    """
    launch_csv_handling.py -c csv -j json
    """
}