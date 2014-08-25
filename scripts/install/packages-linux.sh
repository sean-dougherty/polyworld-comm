#!/bin/bash

function install() {
    if dpkg -s $1 > /dev/null 2>&1 ; then
	return
    fi

    sudo apt-get --assume-yes install $* || exit 1
}

packages="\
	g++ \
	libgsl0-dev \
	gnuplot \
	zlib1g-dev \
	python-scipy \
    mesa-common-dev \
    libglu1-mesa-dev \
    nvidia-cuda-toolkit
"

for pkg in $packages; do
    install $pkg
done
