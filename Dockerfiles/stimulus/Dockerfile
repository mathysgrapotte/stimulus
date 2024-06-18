# python 3.11.8-slim-bullseye intes64
FROM python@sha256:a2d01031695ff170831430810ee30dd06d8413b08f72ad978b43fd10daa6b86e
LABEL maintainer="Alessio Vignoli" \
        name="alessiovignoli3/stimulus:simulus_v{version_number}" \
        description="Docker image containing python packages required for stimulus using modules"

# installing procps outside of python
RUN apt-get update -y\
        && apt-get install -y procps

# installing python needed packages
# TODO one can add the requirements.txt and install the packages from there
RUN pip install \
    numpy==1.26.0 \
    polars-lts-cpu==0.20.30 \
    scikit-learn==1.5.0 \
    torch==2.3.0 \
    ray[train,tune,default]==2.23.0 \
    matplotlib==3.9.0 \
    pandas==2.2.0
