#!/usr/bin/env bash

source env/bin/activate

count=3
until python3 FaceRec.py; do
    if [ $count -eq 0 ]; then
        echo "ERROR: To many restart attempts, stopping"
        exit 1;
    else
        count=$((count-1))
        echo "FaceRec crashed. Respawning, left $count attempts, after 2 seconds.." >&2
        sleep 2
    fi
done


