#!/bin/bash

GITROOT=${1:-${HOME}/bet}
GITBRANCH=${2:-dev}
GITBRANCH=$(basename $GITBRANCH)
echo "Pulling from origin ..."
cd ${GITROOT}
git checkout -B $GITBRANCH && git pull origin $GITBRANCH
