#!/bin/bash

MYSQL_DBDIR='db/mysql'
POSTGIS_DBDIR="db/postgis"
DBPWD="CAMBIAMI"
PHPMYADMINPORT=3307
POSTGISPORT=5432

CHECK_EXIST() {
 docker ps -a --format "{{.ID}}\t{{.Names}}" | grep $1 >/dev/null
 return $?
}

CHECK_RUN() {
 docker ps --format "{{.ID}}\t{{.Names}}" | grep $1 >/dev/null
 return $?
}


if [ ! -d "$MYSQL_DBDIR" ]; then
  echo -n "Dir $MYSQL_DBDIR not found."
  echo "Make sure to run this script from the project root dir!"
  exit 255
else
  MYDBDIR=$(readlink -f  ${MYSQL_DBDIR})
fi

if [ ! -d "$POSTGIS_DBDIR" ]; then
  echo -n "Dir $POSTGIS_DBDIR not found."
  echo "Make sure to run this script from the project root dir!"
  exit 255
else
  POSTDBDIR=$(readlink -f  ${POSTGIS_DBDIR})
fi




NAME="bet-mysql"
if ! CHECK_RUN "$NAME"; then
  if CHECK_EXIST "$NAME" ; then
    echo "Starting container $NAME..."
    docker start $NAME
  else
    echo "Creating container $NAME..."
    echo "docker run --name $NAME -v ${MYDBDIR}:/var/lib/mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=${DBPWD} -d mysql"
  fi
else
   echo "Already running $NAME"
fi

NAME="bet-phpmyadmin"
if ! CHECK_RUN "$NAME"; then
  if CHECK_EXIST "$NAME" ; then
    echo "Starting container $NAME..."
    docker start $NAME
  else
    echo "Creating container $NAME..."
    docker run --name $NAME --link bet-mysql:mysql -p ${PHPMYADMINPORT}:80 -d nazarpc/phpmyadmin
  fi
else
   echo "Already running $NAME"
fi


NAME="bet-postgis"
if ! CHECK_RUN "$NAME"; then
  if CHECK_EXIST "$NAME" ; then
    echo "Starting container $NAME..."
    docker start $NAME
  else
    echo "Creating container $NAME..."
    docker run --name $NAME -v ${POSTDBDIR}:/var/lib/postgresql/data -p ${POSTGISPORT}:5432 -e POSTGRES_PASSWORD=${DBPWD} -d mdillon/postgis
  fi
else
   echo "Already running $NAME"
fi

exit



