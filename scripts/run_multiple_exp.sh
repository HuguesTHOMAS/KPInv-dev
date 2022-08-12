#!/bin/bash

########
# Init #
########

echo ""
echo "Running pytorch docker container"
echo ""

# training script used


# List of arguments for our experiments
"--weight_decay 0.00001"



# for ARGS in "-g | -b" "-fg | -bs" "-fg | -bi"
for ARGS in "-fg | -bs"
do


    ./run_in_container.sh -d -c "python3 experiments/S3DIS_simple/train_S3DIS_simple.py"




    # Read simu and nav params
    IFS="|" read SIMU_ARGS NAV_ARGS <<< $ARGS

    # Start exp
    ./run_in_melodic.sh -d -c "./simu_master.sh $SIMU_ARGS -t 2022-A -p $PARAMS"
    sleep 5.0
    sleep 5.0
    sleep 5.0
    ./run_in_foxy.sh -d -c "./nav_master.sh $NAV_ARGS -m 2"
            
    # Wait for the docker containers to be stopped
    sleep 5.0
    docker_msg=$(docker ps | grep "hth-foxy")
    until [[ ! -n "$docker_msg" ]]
    do 
        sleep 5.0
        docker_msg=$(docker ps | grep "hth-foxy")
        echo "Recieved docker message, continue experiment"
    done 

    # Sleep a bit to be sure  
    echo "Experiment finished"
    sleep 5.0

    if [ "$low_weight_exp" = true ] ; then
        ./run_in_pytorch.sh -d -c "./ros_python.sh clean_last_simu.py"
    fi
    
    sleep 5.0
    
done