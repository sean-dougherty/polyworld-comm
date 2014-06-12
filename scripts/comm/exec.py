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
batch_size = 50
min_success_rate = 0.9
min_success_velocity = 0.5
max_success_std = 0.05
max_steps = 500
repeat = "5,4,3,2"

def sh(cmd):
	rc = os.system(cmd)
	if rc != 0:
		raise Exception("Failed executing cmd. rc=%d, cmd=%s" % (rc, cmd))

def run(id):
	sh('echo %d > trial_seed.txt' % id)
	sh('./Polyworld '+trialsdir+'/trial'+str(id)+'.wf')
	success = get_success()
	score = get_score()
	velocity = get_velocity()
	sh('mv run '+trialsdir+'/run'+str(id))

	return success, score, velocity

def get_success():
	return open('run/trials_result.txt').readline().strip() == 'success'

def get_score():
	return float(open('run/genome/Fittest/fitness.txt').readline().strip().split()[1])

def get_velocity():
	fields = map(float, open('run/velocity.txt').readline().split())
	return {'mean': fields[0], 'stddev': fields[1]}

if i == None:
	#
	# Initial Run
	#
	sh('echo 0 > training_trials_per_patch.txt')
	i = 0
	
	food_difficulty=0
	mutation_rate=' --high-mutation '
	
	if extend_rundir != None:
		sh('scripts/genomeSeed --repeat "'+repeat+'" --fittest '+extend_rundir)
		sh('scripts/comm/mkbranch.py --seed-from-run --max-steps '+str(max_steps)+' --food-difficulty '+str(food_difficulty)+' > '+trialsdir+'/trial0.wf')
	else:
		sh('scripts/comm/mkbranch.py --max-steps '+str(max_steps)+' --food-difficulty 0 > '+trialsdir+'/trial0.wf')
		
	run(i)
else:
	mutation_rate=' --high-mutation '
	food_difficulty = 0
	print "!!!!!!!!!!!!!!!"
	print "!!! WARNING !!! FORCING -1 FOOD DIFFICULTY FOR RESUME"
	print "!!!!!!!!!!!!!!!"
	print "!!!!!!!!!!!!!!!"
	print "!!! WARNING !!! FORCING HIGH MUTATE FOR RESUME"
	print "!!!!!!!!!!!!!!!"

#
# Trials
#
difficulty_log = open(trialsdir + '/difficulty.log', 'a')
score_log = open(trialsdir + '/score.log', 'a')

while True:
	i += 1

	print """\
========================================
i: %d
food difficulty: %d
========================================
""" % (i, food_difficulty)
	
	sh('scripts/genomeSeed --repeat "'+repeat+'" --fittest '+trialsdir+'/run'+str(i-1))
	sh('scripts/comm/mkbranch.py --max-steps '+str(max_steps)+mutation_rate+' --food-difficulty '+str(food_difficulty)+' --seed-from-run > '+trialsdir+'/trial'+str(i)+'.wf')

	success, score, velocity = run(i)

	if velocity['mean'] > 0.5:
		sh('echo 20 > training_trials_per_patch.txt')

	if velocity['mean'] < 0.9:
		mutation_rate = ' --high-mutation '
	elif velocity['stddev'] > 0.1:
		mutation_rate = ' --med-mutation '
	else:
		mutation_rate = ' --low-mutation '

	print 'velocity %f %f --> %s' % (velocity['mean'], velocity['stddev'], mutation_rate)

	difficulty_log.write("%d\t%d\n" % (i, food_difficulty))
	difficulty_log.flush()
	
	score_log.write('%d\t%f\n' % (i, score))
	score_log.flush()

	print """\
  ----------------------------------------
  success: %s
  score: %f
  ----------------------------------------
""" % (success, score)

	#if success:
	#	food_difficulty += 1
