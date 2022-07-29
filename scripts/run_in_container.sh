#!/bin/bash

########
# Init #
########

echo ""
echo "Running pytorch docker container"
echo ""

detach=false
devdoc=false
command=""

while getopts dvc: option
do
case "${option}"
in
d) detach=true;; 
v) devdoc=true;; 
c) command=${OPTARG};;
esac
done

# A developpement container is always detached
if [ "$devdoc" = true ] ; then
    detach=true
fi

# Path to the Network result folder
RES_FOLDER="$PWD/../results"

##########################
# Start docker container #
##########################

# Docker run arguments
docker_args="-it --rm --shm-size=64g "

# Running on gpu (Uncomment to enable gpu)
docker_args="${docker_args} --gpus all "

# Docker run arguments (depending if we run detached or not)
now=`date +%Y-%m-%d_%H-%M-%S`
if [ "$detach" = true ] ; then
    docker_args="-d ${docker_args}"
fi

# Volumes (modify with your own path here)
volumes="-v $PWD/..:/home/$USER/KPInv-dev \
-v $PWD/../../../Data:/home/$USER/Data"

# Additional arguments to be able to open GUI
XSOCK=/tmp/.X11-unix
XAUTH=/home/$USER/.Xauthority
other_args="-v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    --net=host \
    --privileged \
	-e XAUTHORITY=${XAUTH} \
    -e DISPLAY=$DISPLAY \
    -w /home/$USER/KPInv-dev"

if [ "$devdoc" = true ] ; then

    # This is to create a developpement container. No command is run
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "dev-KPInv" \
    kpinv_$USER \
    $command

else

    if [ "$detach" = true ] ; then
        mkdir -p $RES_FOLDER/Log_"$now"
    fi

    # python command started in the docker
    if [ ! "$command" ] ; then
        command="python3 train_S3DIS_simple.py"
    fi

    # Adding detached folder as command argument if needed
    if [ "$detach" = true ] ; then
        if [[ $command == *"python3 train_"* ]]; then
            command="$command Log_$now"
        fi
    fi

    echo -e "Running command $command\n"

    # Execute the command in docker (Example of command: ./master.sh -ve -m 2 -p Sc1_params -t A_tour)
    docker run $docker_args \
    $volumes \
    $other_args \
    --name "$USER-KPInv-$now" \
    kpinv_$USER \
    $command

    # Attach a log parameters and log the detached docker
    if [ "$detach" = true ] ; then
        docker logs -f "$USER-SOGM-$now" &> $RES_FOLDER/Log_"$now"/log.txt &
    fi


fi