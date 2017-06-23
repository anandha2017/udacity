# Use an official Python runtime as a base image
FROM continuumio/anaconda3

MAINTAINER Anandha Ponnampalam <docker2017@ponnampalam.ca>

# Lesson 10, all other pre-requisites are already installed in anaconda3 image above. Only missing below two
RUN pip install tensorflow
RUN pip install TFLearn

VOLUME  "/notebooks"
WORKDIR "/opt/notebooks"

CMD ["/bin/bash"]
