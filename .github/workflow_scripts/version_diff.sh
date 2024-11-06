#!/bin/bash

# List files in the specified S3 bucket and folder
files=$(aws s3 ls s3://autogluon-ci/package_versions/)

# Extract the filenames with the pattern package_versions_{datetimestamp}.txt
latest_file=""
latest_timestamp=0
while read -r line; do
    filename=$(echo $line | awk '{print $4}')
    timestamp=$(echo $filename | cut -d'_' -f3 | cut -d'.' -f1)
    if [ -n "$timestamp" ]; then
        formatted_timestamp=$(echo $timestamp | sed 's/-/ /3')
        # Convert to UNIX format
        timestamp_seconds=$(date -d "$formatted_timestamp" +%s)
    else
        timestamp_seconds=0
    fi
    
    if [ $timestamp_seconds -gt $latest_timestamp ]; then
        latest_timestamp=$timestamp_seconds
        latest_file=$filename
    fi
done <<< "$files"

aws s3 cp s3://autogluon-ci/package_versions/$latest_file ./
old_latest_file="old_${latest_file}"
mv "./$latest_file" "./$old_latest_file"

diff ./package_versions_* ./old_package_versions_* > ./diff_output.txt
diff_exit_code=$?

if [ $diff_exit_code -eq 0 ]; then
    echo "No difference"
elif [ $diff_exit_code -eq 1 ]; then
    echo -e "\nPackage Differences Below:\n"
    # Create arrays to store Name:Version for ordering and matching
    declare -A prev_packages
    declare -A curr_packages

    while IFS= read -r line; do 

        if [[ $line == *"-e git+https:"* ]] && [[ $line == *"autogluon"* ]] || ! [[ $line =~ ^[\<\>] ]] || [[ $line == *"Timestamp:"* ]] ; then
            continue
        fi
        
        if [[ $line == \<* ]]; then
            name=$(echo "$line" | cut -d= -f1 | cut -d' ' -f2)
            version=$(echo "$line" | cut -d= -f2-)
            curr_packages[$name]=$version
        fi
        
        if [[ $line == \>* ]]; then
            name=$(echo "$line" | cut -d= -f1 | cut -d' ' -f2)
            version=$(echo "$line" | cut -d= -f2-)
            prev_packages[$name]=$version
        fi
    done < ./diff_output.txt

    # Create table
    echo "| Previous CI Run | Current CI Run |" > table_output.txt
    echo "| :---: | :---: |" >> table_output.txt
    for key in "${!prev_packages[@]}" "${!curr_packages[@]}"; do
        prev="${key}=${prev_packages[$key]}"
        curr="${key}=${curr_packages[$key]}"
        if [[ -z ${prev_packages[$key]} ]]; then
            prev="-"
        fi
        if [[ -z ${curr_packages[$key]} ]]; then
            curr="-"
        fi
        echo "| $prev | $curr |" >> table_output.txt
    done

    cat ./diff_output.txt
else
    echo "Error: diff command failed with exit code $diff_exit_code"
    exit 1
fi
