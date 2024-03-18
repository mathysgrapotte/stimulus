# python 3.11.8-slim-bullseye
FROM python@sha256:a2d01031695ff170831430810ee30dd06d8413b08f72ad978b43fd10daa6b86e
LABEL maintainer="Alessio Vignoli" \
        name="alessiovignoli3/stimulus:torch_scikit_numpy" \
        description="Docker image containing python packages required for model-check pipeline"

# installing python needed packages
# TODO one can add the requirements.txt and install the packages from there
RUN pip install \
    numpy==1.26.0 \
    pytorch-lightning==2.0.1 \
    scikit-learn==1.3.0
