#!/bin/bash
declare -i count=1
for d in */ ; do
    cd $d
    find . -type f -exec printf '%i\n' $count \;
    count=$((count+1))
    cd ..
done
