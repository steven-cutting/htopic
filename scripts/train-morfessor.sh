#! /usr/bin/env bash

                                                                           # model.bin  traindata.csv
echo "morfessor-train --encoding=utf-8 --traindata-list --logfile=log.log -s ${1} ${2}" | bash +x
