#!/bin/bash

#docker run --rm -it --device=/dev/video0 -p 8080:8080 --network="host" -v $(pwd):/app qualityinspection:latest

python3 ./app.py -tu='192.168.29.146:8001' -s=1
