#!/usr/bin/python

import numpy
import os
import sys
import threading
import time

args = sys.argv[1:]

def err(msg):
	sys.stderr.write(msg)
	sys.stderr.write('\n')
	exit(1)

if len(args) != 1:
	err("usage: exec <trialsdir>")

trialsdir = args[0]

if os.path.exists(trialsdir):
	err('%s already exists!' % trialsdir)

def sh(cmd):
	rc = os.system(cmd)
	if rc != 0:
		raise Exception("Failed executing cmd. rc=%d, cmd=%s" % (rc, cmd))

if os.path.exists('run'):
	sh('mv run run_%d' % int(time.time()))

def run():
	sh('scripts/comm/mkvat.py > trials.wf')
	sh('./Polyworld trials.wf')

run_thread = threading.Thread(target = run)
run_thread.start()

while not os.path.exists('run'):
	time.sleep(1)

sh('ln -s run %s' % trialsdir)
sh('echo "%s" > run/trials-name.txt' % trialsdir)

while True:
	cmd = raw_input("Enter command: ")
	if cmd == 'stop':
		sh('touch run/stop')
		run_thread.join()
		sh('rm %s' % trialsdir)
		sh('mv run %s' % trialsdir)
		sh('rm trials.wf')
		break
	else:
		print 'Invalid command.'
