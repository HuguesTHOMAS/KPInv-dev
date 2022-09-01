#!/bin/bash


########################################################################
# Get the input command #
########################################################################

detach=false
command=""

while getopts dvc: option
do
case "${option}"
in
d) detach=true;; 
c) command=${OPTARG};;
esac
done



########################################################################
# Automatically find an available GPU and start the given script on it #
########################################################################


# Define you environment GPUs
num_gpu="2"
gpu_ids=($(for ((i=0;i<$num_gpu;i+=1)); do echo "${i}"; done))
used_gpus=( )

# Get nvidia-smi string
smi_msg=$(nvidia-smi | grep -A 10 "| Processes: ")
smi_msg2=$(grep -A 10 "|=" <<< "$smi_msg")

# Loop on nvidia-smi lines to get the used GPUs
while IFS= read -r line; do
    # Ignore useless first and last lines 
    if  [[ $line == "| "* ]] ;
    then
        # Ignore graphical processes
        if  [[ $line == *" C "* ]] ;
        then
            used_gpu=$(echo $line | cut -d' ' -f2)
            used_gpus=(${used_gpus[@]} "$used_gpu")
        fi
    fi
done <<< "$smi_msg2"

# Get the first available GPU
chosen_gpu=""
for gpu_id in ${gpu_ids[@]}; do
    if [[ ! " ${used_gpus[*]} " =~ " ${gpu_id} " ]]; then
        chosen_gpu=${gpu_id}
    fi
done


if [[ ! "$chosen_gpu" ]]; 
then
    # If there is no GPU available stop
    echo " "
    echo "No GPU available"
    echo "Abort experiement"
    echo " "

else
    echo " "
    echo " +------------------------------------+"
    echo " | Starting a new experiment on GPU $chosen_gpu |"
    echo " +------------------------------------+"
    echo " "
    
    # Start 
    args="-g $chosen_gpu"
    if [ "$detach" = true ] ; then
        args="-d ${args}"
    fi
    ./run_in_container.sh $args -c "$command"
fi







