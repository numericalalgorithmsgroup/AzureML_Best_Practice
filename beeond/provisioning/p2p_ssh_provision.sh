#!/bin/bash

sudo bash <<"EOC"
if [ ! -f /root/.ssh/id_rsa ]; then
    ssh-keygen -b 4096 -f /root/.ssh/id_rsa -N ""
fi
EOC

sudo cat /root/.ssh/id_rsa.pub | tee $HOME/masterkey
