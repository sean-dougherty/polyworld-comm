#!/bin/bash

if [ "$1" != "--field" ]; then
    __pwfarm_script.sh --output foo /tmp/test_output $0 --field $*
else
    echo "ARGS:"
    echo $*
    echo
    echo "PWD:"
    pwd
    date

    echo hi > a;
    mkdir bye
    echo foo > bye/b
    zip -r $PWFARM_OUTPUT_FILE .

    #read -p "[test] press enter..."
    sleep 3
    exit 1
fi