#!/bin/bash

sudo apt-get update
sudo apt-get install htop
sudo apt-get -y install docker.io
sudo usermod -aG docker $(whoami)