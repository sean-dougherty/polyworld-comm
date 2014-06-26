#!/usr/bin/python

import sys

InitSeed = 10 # RNG Seed 

MutationRate = 0.005
MaxAgents = 1000
NumberFittest = 0

args = sys.argv[1:]
while True:
	if len(args) and args[0][:2] == '--':
		sys.stderr.write('invalid option: %s\n' % args[0])
		exit(1)
	else:
		break

min_mutation = MutationRate
max_mutation = float(str(MutationRate) + "1")

print '@version 2'

print 'InitSeed', InitSeed
print 'MinMutationRate', min_mutation
print 'MaxMutationRate', max_mutation
print 'NumberFittest', NumberFittest

print """\
MinAgents 1
MaxAgents %d
InitAgents %d
SeedAgents %d
""" % (MaxAgents, MaxAgents, MaxAgents)

print """\
Vision False

MinCrossoverPoints 1
MaxCrossoverPoints 4

MinLifeSpan 100000
MaxLifeSpan 100001

EnableTopologicalDistortionRngSeed True
EnableInitWeightRngSeed True

EnableHearing True
EnableVoice True
NumSoundFrequencies 2

MinAgentMaxEnergy 1000.0
MaxAgentMaxEnergy 1000.1
EnergyUseMultiplier 0.001
"""
