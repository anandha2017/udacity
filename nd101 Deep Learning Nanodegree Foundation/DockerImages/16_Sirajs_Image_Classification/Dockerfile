# Use an official Python runtime as a base image
FROM continuumio/anaconda3

MAINTAINER Anandha Ponnampalam <docker2017@ponnampalam.ca>

RUN pip install tensorflow
RUN pip install TFLearn
RUN pip install keras
RUN pip install pillow

VOLUME  "/notebooks"
WORKDIR "/opt/notebooks"

CMD ["/bin/bash"]
