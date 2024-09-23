# python 3.11.8-slim-bullseye intes64
FROM python@sha256:a2d01031695ff170831430810ee30dd06d8413b08f72ad978b43fd10daa6b86e
LABEL maintainer="Alessio Vignoli" \
        name="alessiovignoli3/stimulus:simulus_v{version_number}" \
        description="Docker image containing python packages required for stimulus using modules"

# installing procps outside of python
RUN apt-get update -y\
        && apt-get install -y procps

# installing python needed packages
RUN pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple stimulus-py==0.0.8
