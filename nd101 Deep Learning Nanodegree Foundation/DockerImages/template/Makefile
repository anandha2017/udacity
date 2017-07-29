help:
	@cat Makefile

SRC=$(shell pwd)

build:
	docker build -t myapp .

bash: build
	docker run -it -v "$(SRC)/notebooks":/opt/notebooks myapp /bin/bash

notebook: build
	docker run -it \
	  -v "$(SRC)/notebooks":/opt/notebooks \
	  -p 8888:8888 myapp /bin/bash -c \
	  "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"

tensor: build
	docker run -it \
	  -v "$(SRC)/notebooks":/opt/notebooks \
	  -e KERAS_BACKEND=tensorflow \
	  -p 8888:8888 myapp /bin/bash -c \
	  "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"

theano: build
	docker run -it \
	  -v "$(SRC)/notebooks":/opt/notebooks \
	  -e KERAS_BACKEND=theano \
	  -p 8888:8888 myapp /bin/bash -c \
	  "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
