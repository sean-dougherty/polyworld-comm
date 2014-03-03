# About polyworld-comm

This project is a fork of https://sourceforge.net/projects/polyworld/.
The purpose of this fork is to allow for experimentation with emergence of
structured communication. Any useful results will hopefully be pushed back to
the original project.

## System Requirements
Ubuntu 12.04 or higher

# Building/Installing

Instructions in "code boxes" are to be executed in a terminal.

## Download Source

### Install Git
If you don't already have git, install it via:
```
sudo apt-get install git
```

### Download Source
Download the source with the git command. This will create a directory named
_polyworld-comm_ under your current directory. If, for example, you execute
this command from your home directory (/home/suzy), then the source will be in
/home/suzy/polyworld-comm.
```
git clone https://github.com/sean-dougherty/polyworld-comm.git
```

## Build
Go to the _polyworld-comm_ directory you just downloaded:
```
cd polyworld-comm
```

### Install Dependencies
Polyworld requires that a number of packages be installed. If you're using a
supported version of Ubuntu then you can simply do the following to install
them:
```
scripts/install/packages-linux.sh
```

### Compile
It's finally time to compile the source! Issue the following command:
```
make
```
If it worked then the file you should see a message like
"scons: done building targets." and ./Polyworld should exist,
which you can verify via:
```
ls ./Polyworld
```

## Verify Polyworld Runs
Now, let's see if Polyworld runs on your machine. Execute the following:
```
./Polyworld worldfiles/hello_world.wf
```

