help:
	@cat Makefile

SRC=$(shell pwd)

build:
	docker build -t myapp .

bash: build
	docker run -it -v "$(SRC)/notebooks":/opt/notebooks myapp bash

jupyter: build
	docker run -it \
	  -v "$(SRC)/notebooks":/opt/notebooks \
	  -p 8888:8888 myapp /bin/bash -c \
	  "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
