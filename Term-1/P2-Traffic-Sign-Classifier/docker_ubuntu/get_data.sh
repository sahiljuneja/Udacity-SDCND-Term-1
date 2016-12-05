#!/bin/bash

curl -LOk https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip

sudo apt-get install unzip
unzip traffic-sign-data.zip
sudo mv lab\ 2\ data/ /home/ubuntu/Udacity-Self-Driving-Car-ND/Term-1/P2-Traffic-Sign-Classifier/

