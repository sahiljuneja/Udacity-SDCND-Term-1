#!/bin/bash

curl -LOk https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip

sudo apt-get install unzip
unzip traffic-sign-data.zip
mv lab\ 2\ data/ lab2_data

