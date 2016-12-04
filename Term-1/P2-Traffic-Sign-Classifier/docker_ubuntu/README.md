### Still testing the following

### Reference	

Thanks to Hasan Korre for these files https://github.com/hkorre/traffic-signs/tree/master/docker_ubuntu

### Overview

This folder consists of scripts to download, install and run a docker container for Project 2 in an AWS instance (Non-GPU).

### Instructions
Please note that the following instance will cost you, and I have only tried with the following instance. Any further experimentation is on you.

1. Launch AWS m4.2xlarge Instance (ami-01f05461) with appropriate Security Group (I used https://gist.github.com/iamatypeofwalrus/5183133 for reference). 
 
2. Once Instance is launched, SSH into it. I use Putty. Refer to http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html for details.

3. Run the following after SSHing
	* $ sudo apt-get install git
	* git clone https://github.com/yourHandle/whereYouSavedYourProjectFiles.git
	* grab https://github.com/sahiljuneja/Udacity-Self-Driving-Car-ND/tree/master/Term-1/P2-Traffic-Sign-Classifier/docker_ubuntu

4. Build Docker Container
	* $ cd docker_ubuntu (wherever cloned/downloaded)
	* $ sudo ./install.sh
	* $ sudo ./build.sh

5. Edit run.sh and Run the Container
	* In run.sh modify the line "-v /home/ubuntu/Udacity-Self-Driving-Car-ND:/home/Udacity-Self-Driving-Car-ND \ " to correspond to your own cloned project repo directory
	* $ sudo ./run.sh

6. Run notebook
	* jupyter notebook --ip='*' --port=8888 --no-browser

7. Access notebook in your local machine browser
	* public_ip:8888 where public_ip is the ip of your aws instance

Don't forget to terminate your instance once you are done.

