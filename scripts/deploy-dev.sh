#!/bin/bash

GITROOT=${1:-${HOME}/bet}
echo "Pulling from origin ..."
cd ${GITROOT}
git checkout dev && git pull origin dev
