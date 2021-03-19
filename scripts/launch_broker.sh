#!/bin/bash


CHECK_EXIST() {
 docker ps -a --format "{{.ID}}\t{{.Names}}" | grep $1 >/dev/null
 return $?
}

CHECK_RUN() {
 docker ps --format "{{.ID}}\t{{.Names}}" | grep $1 >/dev/null
 return $?
}


NAME="bet-rabbitmq"
if ! CHECK_RUN "$NAME"; then
  if CHECK_EXIST "$NAME" ; then
    echo "Starting container $NAME..."
    docker start $NAME
  else
    echo "Creating container $NAME..."
    docker run --name $NAME -p 5672:5672 -d rabbitmq
  fi
else
   echo "Already running $NAME"
fi



