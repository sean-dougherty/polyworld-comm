#!/usr/bin/python

import sys

InitSeed = 1 # RNG Seed

InitAgents = 40 # Total number of random/seed agents in generation 0
SeedAgents0 = 0 # Number of seed agents in generation 0
SeedAgents =  0 # Number of seed agents in generations 1..N
MutationRate = 0.02
NumberFittest = 30

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
print 'InitAgents', InitAgents
print 'SeedAgents0', SeedAgents0
print 'SeedAgents', SeedAgents
print 'MinMutationRate', min_mutation
print 'MaxMutationRate', max_mutation
print 'NumberFittest', NumberFittest

print """\
Vision False

MinAgents 1
MaxAgents 150
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
