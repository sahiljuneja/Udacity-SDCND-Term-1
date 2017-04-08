docker run --rm \
       -p 8888:8888 \
       -v /home/ubuntu/Udacity-Self-Driving-Car-ND:/home/Udacity-Self-Driving-Car-ND \
       -w /home \
       --name sj_carnd \
       -h sj-carnd \
       -it sjuneja_trusty:carnd \
       /bin/bash
