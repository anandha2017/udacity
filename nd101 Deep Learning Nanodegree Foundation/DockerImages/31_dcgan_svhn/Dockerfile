FROM continuumio/anaconda3

MAINTAINER Anandha Ponnampalam <docker2017@ponnampalam.ca>

RUN conda update setuptools
RUN conda install tqdm
RUN pip install tensorflow
RUN pip install TFLearn

VOLUME  "/notebooks"
WORKDIR "/opt/notebooks"

CMD ["/bin/bash"]
