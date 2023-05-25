#!/bin/bash
#
set -xue

eval_path="/local-scratch1/data/qkun/semafor/Example_Probeset/3.3.4_MMA_Inconsistencies"

for type in ${eval_path}/*; do
    for id in "${type}"/*; do
        cd "${id}"
        tar -xvf *.tar.gz
    done
done

train_path="/local-scratch1/data/qkun/semafor/semafor_data_collection"
cd $train_path
for file in ./*.tar.gz; do
    tar -xvf $file
    rm -f $file
done
