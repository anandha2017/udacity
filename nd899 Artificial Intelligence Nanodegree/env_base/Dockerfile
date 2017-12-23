FROM continuumio/anaconda3

MAINTAINER Anandha Ponnampalam <docker2017@ponnampalam.ca>

# These commands run when a new image is built
ONBUILD ADD .condarc /root/.condarc
ONBUILD ADD environment.yml /environment.yml
ONBUILD RUN conda env create -f /environment.yml -n app
ONBUILD ADD . /app

ENTRYPOINT ["/opt/conda/envs/app/bin/python"]
