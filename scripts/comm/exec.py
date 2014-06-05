#!/usr/bin/python

import numpy
import os
import sys

args = sys.argv[1:]

trialsdir = args[0]
batch_size = 50
min_success_rate = 0.9
min_success_velocity = 0.5
max_success_std = 0.18
max_steps = 1000
repeat = "5,4,3,2"

def sh(cmd):
	rc = os.system(cmd)
	if rc != 0:
		raise Exception("Failed executing cmd. rc=%d, cmd=%s" % (rc, cmd))

times = []
velocities = []

def run(id):
	global times
	sh('./Polyworld '+trialsdir+'/trial'+str(id)+'.wf')
	t = get_run_time()
	d = get_food_distance()
	times.append(t)
	velocities.append(d / t)
	sh('mv run '+trialsdir+'/run'+str(id))

def get_run_time():
	return int(open('run/endStep.txt').readline())

def get_food_distance():
	return float(open('run/fooddist.txt').readline())

#
# Initial Run
#
i = 0
sh('scripts/comm/mkbranch.py --max-steps '+str(max_steps)+' --food-difficulty 0 --random-single-food 0 > '+trialsdir+'/trial0.wf')
run(i)

#
# Trials
#
food_difficulty=0

class Batch:
	start_id=1
	success_count=0
	attempt_count=0

	@staticmethod
	def std():
		t = velocities[Batch.start_id:]
		assert len(t) <= batch_size
		return numpy.std(t)

	@staticmethod
	def mean():
		return numpy.mean(velocities[Batch.start_id:])

velocity_log = open(trialsdir + '/velocity.log', 'w')

while True:
	i += 1
	Batch.attempt_count += 1

	print """\
========================================
i: %d
food difficulty: %d
batch start: %d
batch attempts: %d
batch successes: %d
batch mean: %f
batch stddev: %f
========================================
""" % (i, food_difficulty, Batch.start_id, Batch.attempt_count - 1, Batch.success_count, Batch.mean(), Batch.std())
	
	sh('scripts/genomeSeed --repeat "'+repeat+'" --fittest '+trialsdir+'/run'+str(i-1))
	sh('scripts/comm/mkbranch.py --max-steps '+str(max_steps)+' --food-difficulty '+str(food_difficulty)+' --random-single-food '+str(i)+' --seed-from-run > '+trialsdir+'/trial'+str(i)+'.wf')
	run(i)

	t = times[-1]
	v = velocities[-1]

	velocity_log.write('%d\t%f\n' % (i, v))
	velocity_log.flush()

	print """\
  ----------------------------------------
  t: %d
  food dist: %f
  velocity: %f
  ----------------------------------------
""" % (t, t*v, v)

	is_success = v > min_success_velocity
	if is_success:
		Batch.success_count += 1
	
	if Batch.attempt_count == batch_size:
		success_rate = float(Batch.success_count) / batch_size
		mean = Batch.mean()
		std = Batch.std()

		print """\
========================================
END OF BATCH
success_rate: %f
mean: %f
stddev: %f
========================================
""" % (success_rate, mean, std)

		if success_rate < min_success_rate:
			print "REJECTED DUE TO SUCCESS RATE"
			batch_successful = False
		elif std > max_success_std:
			print "REJECTED DUE TO STDDEV"
			batch_successful = False
		else:
			print "ACCEPTED!"
			batch_successful = True
			if food_difficulty < 2:
				food_difficulty += 1


		Batch.attempt_count = 0
		Batch.success_count = 0
		Batch.start_id += batch_size
