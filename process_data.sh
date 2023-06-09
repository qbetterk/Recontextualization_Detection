#!/bin/bash
#
set -xue

# eval_path="/local-scratch1/data/qkun/semafor/Example_Probeset/3.3.4_MMA_Inconsistencies"

# for type in ${eval_path}/*; do
#     for id in "${type}"/*; do
#         cd "${id}"
#         tar -xvf *.tar.gz
#     done
# done

# train_path="/local-scratch1/data/qkun/semafor/semafor_data_collection"
# cd $train_path
# for file in ./*.tar.gz; do
#     tar -xvf $file
#     rm -f $file
# done

eval_path="./4.1.2/Demoralize"
for article in "${eval_path}"/*; do
    cd "$article"
    mkdir extract -p
    for twitter in *; do
        if [ "$twitter" = "reference_article" ]; then
            tar -xvf "$twitter"/*/*.tar.gz -C extract/
        elif [ "$twitter" != "extract" ]; then
            tar -xvf "${twitter}"/*.tar.gz -C extract/
        fi
    done
    cd ../../..
done