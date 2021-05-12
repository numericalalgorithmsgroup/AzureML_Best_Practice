#!/bin/bash

mkdir /ebbuild
cd /ebbuild


apt-get install -y build-essential debhelper devscripts fakeroot git libaio-dev \
        libboost-filesystem-dev libboost-program-options-dev libboost-thread-dev \
        libncurses-dev libnuma-dev lintian
git clone https://github.com/breuner/elbencho.git .

make -j8

make deb

apt-get install ./packaging/elbencho*.deb

cd /
rm -r /ebbuild
