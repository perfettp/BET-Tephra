#!/bin/sh

echo "/usr/bin/gdalwarp -t_srs '+proj=longlat +datum=WGS84 +no_defs' $1 $2"
/usr/bin/gdalwarp -t_srs '+proj=longlat +datum=WGS84 +no_defs' $1 $2
