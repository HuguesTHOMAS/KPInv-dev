#!/bin/bash

########
# Init #
########

# Number of docker allowed to work in parrallel
# *********************************************

max_containers="2"


# Training script used
# ********************

training_script="experiments/ScanObjectNN/train.py"
# training_script="experiments/S3DIS_simple/train_S3DIS_simple.py"


# Declare an array variable
# *************************

declare -a arg_arr=("--inv_groups 8 --layer_blocks 4 4 12 20 4"
                    "--inv_groups 8"
                    "--inv_groups 8 --layer_blocks 4 4 12 20 4"
                    "--inv_groups 8"
                    "--inv_groups 8 --layer_blocks 4 4 12 20 4")


echo ""
echo ""
echo "        +---------------------------------------+"
echo "        |  Multiple experiments running script  |"
echo "        +---------------------------------------+"
echo ""
echo "$max_containers docker containers allowed to run in parallel."
echo "Using training script: $training_script."
echo "The script will be running the following experiments:"
for arg_str in "${arg_arr[@]}"
do
    echo "    $arg_str"
done
echo ""
echo ""

echo "----------------------------------------------------------------------"

for arg_str in "${arg_arr[@]}"
do
    echo ""
    echo "Experiment:    $arg_str"

    # Wait for the other docker containers to be stopped
    echo ""
    echo "Waiting for available other containers to be finished ..."
    docker_msg=$(docker ps | grep "hth-KP")
    n_dockers=$(echo "$docker_msg" | tr " " "\n" | grep -c "hth-KP")
    until [[ "$n_dockers" -lt "$max_containers" ]]
    do 
        sleep 1.0
        docker_msg=$(docker ps | grep "hth-KP")
        n_dockers=$(echo "$docker_msg" | tr " " "\n" | grep -c "hth-KP")
    done 
    echo ""
    echo "Found only $n_dockers docker container running. Starting a new one"

    # Wait to be sure GPU is used
    sleep 3.0
    
    # Start 
    ./auto_gpu.sh -d -c "python3 $training_script $arg_str"

    # Wait to be sure GPU is used
    sleep 3.0
    
    echo ""
    echo "----------------------------------------------------------------------"
    
done