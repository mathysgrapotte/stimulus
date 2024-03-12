
process INTERPRET_JSON {

    container "python@sha256:a2d01031695ff170831430810ee30dd06d8413b08f72ad978b43fd10daa6b86e"   // python 3.11.8-slim-bullseye
    label 'process_low'

    input:
    path user_json

    output:
    stdout emit: standardout

    """
    launch_interpret_json.py -j ${user_json}
    """

}