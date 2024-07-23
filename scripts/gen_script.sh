#!/bin/bash

c_option=""
c_suffix=""
path="data/generate_input_example.txt"

while getopts ":c" opt; do
    case $opt in
        c)
            c_option="-c"
            c_suffix="_c"
            path="data/generate_czech_input_example.txt"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Define the modes
modes=("yn" "alt" "tf" "wh" "whmc" "cloze" "clozemc" "mixed")

# Loop through each mode and call the python script
for mode in "${modes[@]}"; do
    output_file="gen_${mode}${c_suffix}.txt"
    python3 src/generation.py -m $mode -f $path $c_option > "examples/$output_file" 2>&1
done

