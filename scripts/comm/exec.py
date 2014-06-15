#!/usr/bin/python

import numpy
import os
import sys

args = sys.argv[1:]

extend_rundir = None
i = None

if args[0] == '--extend':
	extend_rundir = args[1]
	args = args[2:]
elif args[0] == '--resume':
	i = int(args[1])
	args = args[2:]

trialsdir = args[0]
repeat = "5,4,3,2"

def sh(cmd):
	rc = os.system(cmd)
	if rc != 0:
		raise Exception("Failed executing cmd. rc=%d, cmd=%s" % (rc, cmd))

def run(id):
	sh('echo %d > trial_seed.txt' % id)
	sh('./Polyworld '+trialsdir+'/trial'+str(id)+'.wf')
	score = get_score()
	sh('mv run '+trialsdir+'/run'+str(id))

	return score

def get_score():
	return float(open('run/genome/Fittest/fitness.txt').readline().strip().split()[1])

if i == None:
	#
	# Initial Run
	#
	i = 0
	
	food_difficulty=0
	
	if extend_rundir != None:
		sh('scripts/genomeSeed --repeat "'+repeat+'" --fittest '+extend_rundir)
		sh('scripts/comm/mkvat.py --seed-from-run > '+trialsdir+'/trial0.wf')
	else:
		sh('scripts/comm/mkvat.py > '+trialsdir+'/trial0.wf')
		
	run(i)

#
# Trials
#
score_log = open(trialsdir + '/score.log', 'a')

while True:
	i += 1

	print """\
========================================
i: %d
========================================
""" % (i)
	
	sh('scripts/genomeSeed --repeat "'+repeat+'" --fittest '+trialsdir+'/run'+str(i-1))
	sh('scripts/comm/mkvat.py --seed-from-run > '+trialsdir+'/trial'+str(i)+'.wf')

	score = run(i)

	score_log.write('%d\t%f\n' % (i, score))
	score_log.flush()

	print """\
  ----------------------------------------
  score: %f
  ----------------------------------------
""" % (score)
