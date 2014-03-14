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
If it worked then you should see a message like
"scons: done building targets." and ./Polyworld should exist,
which you can verify via:
```
ls ./Polyworld
```

### Verify Polyworld Runs
Now, let's see if Polyworld runs on your machine. Execute the following:
```
./Polyworld worldfiles/hello_world.wf
```
You should see a graphical window appear containing a 3D simulation.

# Collaborating on the Project

## Getting source/worldfile updates from Github

To download updates posted to Github by another author, you should first open
a terminal and then change directories to where you originally downloaded the
source. For example, if you originally downloaded to ~/polyworld-comm, then
you would execute:

```
cd ~/polyworld-comm
```

Next, download the github content:

```
git pull origin master
```

Finally, you should do a build to make sure your binaries are current:

```
make
```

# Designing/Executing Simulations

## Importing agents from another simulation

If you would like to use the genomes of agents evolved in some Simulation A as
the first generation in a Simulation B, then you must:

### 1. Worldfile A: Record genomes/positions

*If another author has already designed the worldfiles of A and B to perform
an import, you can skip this step.*

Simulation A must record its genomes and (optionally) the agent positions. To
record genomes, you must have the following in A's worldfile:

```
RecordGenomes True
```

Or, you can make the simulation record *everything*, including genomes:

```
RecordAll True
```

If you would like to preserve the positions of the agents, which can be
important in simulations that have distinct sub-populations, then you must
have the following in A's worldfile:

```
RecordPosition Precise
```

You may also use the *RecordAll* property to cause the recording of positions.
Note that there is also an *Approximate* setting for the position log, but
that format is not currently supported by the seed code.

### 2. Worldfile B: Seed from run

*Like step 1, you may skip this if another author designed the worldfiles*

*Warning! The agents in Simulation B must have exactly the same genome
schema as in Simulation A. For example, you can't add new internal groups
or a new sensor to Simulation B.*

The worldfile from Simulation B must be modified to specify it is importing
its *seed* genomes from a run. To do so, you must have the following in
Simulation B's worldfile:

```
SeedGenomeFromRun True
```

Next, you must specify the number of seed agents that will be created. For
example, if you want 180 agents from Simulation A:

```
SeedAgents 180
```

If you want to preserve agent position information, then you must include
the following in B's worldfile:

```
SeedPositionFromRun True
```

### 3. Move/rename Simulation A run/

The run directory for Simulation A musn't be the default directory ./run.
You must rename/move it.

If you are using worldfiles from github, it is suggested that you create
a *runs* directory with a structure reflecting that of *worldfiles*. For
example, if your Simulation A is from worldfile
*worldfiles/comm/predator/deaf/0.0-motionless-eating.wf*, then use the
following commands to move its run directory:

```
mkdir -p runs/comm/predator/deaf/0.0-motionless-eating

mv run runs/comm/predator/deaf/0.0-motionless-eating/run_0
```

### 4. Create ./seedGenomes.txt and ./seedPositions.txt

Simulation B requires the presence of the file *./genomeSeeds.txt*, which
can be conveniently generated with the tool *scripts/genomeSeed*. The tool
is capable of several behaviors, which you can learn about by running
the command with no arguments. By default, it will generate a file for
the final generation of agents in the run it is pointed at.

To generate a *seedGenomes.txt* for the final generation of Simulation A,
assuming the run directory is located at
*runs/comm/predator/deaf/0.0-motionless-eating/run_0*, execute the following:

```
./scripts/genomeSeed runs/comm/predator/deaf/0.0-motionless-eating/run_0
```

If would you like to also import agent positions, then you must specify the
*--pos* flag:

```
./scripts/genomeSeed --pos runs/comm/predator/deaf/0.0-motionless-eating/run_0
```

### 5. Execute Simulation B

You're now ready to execute Simulation B. Phew!

If you've done things correctly, you should see a large number of messages
written to stdout when you execute Simulation B indicating that it is seeding
from a previous run. An example line of output follows:

```
seeding agent #1 genome from runs/comm/predator/deaf/0.0-motionless-eating/run_0/genome/agents/genome_184850.txt
```