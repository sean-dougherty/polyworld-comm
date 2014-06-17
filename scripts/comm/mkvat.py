#!/usr/bin/python

import sys

Seed_Agents_run0 = 0
Seed_Agents = 40
seed_from_run = False

mutation_rate = 0.001
high_mutation_rate = 0.05
high_mutation_rate_generation = 100

number_fittest = 30

args = sys.argv[1:]
while True:
	if len(args) and args[0] == '--seed-from-run':
		seed_from_run = True
		args = args[1:]
	elif len(args) and args[0] == '--gen':
		gen = int(args[1])
		if high_mutation_rate_generation != None and gen % high_mutation_rate_generation == 0:
			mutation_rate = high_mutation_rate
		args = args[2:]
	elif len(args) and args[0][:2] == '--':
		sys.stderr.write('invalid option: %s\n' % args[0])
		exit(1)
	else:
		break

if seed_from_run:
	seed_agents = Seed_Agents
else:
	seed_agents = Seed_Agents_run0

min_mutation = mutation_rate
max_mutation = float(str(mutation_rate) + "1")

print """\
@version 2
"""

print """\
MinMutationRate %f
MaxMutationRate %f
""" % (min_mutation, max_mutation)

print """\
NumberFittest %d
""" % number_fittest

print """\
SeedGenomeFromRun %s
SeedAgents %d
""" % (seed_from_run, seed_agents)


print """\
Vision False

InitSeed 1

MinAgents 1
MaxAgents 150
InitAgents 40
SeedMutationProbability 0.5
MinCrossoverPoints 1
MaxCrossoverPoints 4

MinLifeSpan 100000
MaxLifeSpan 100001
MateWait 0

EnableTopologicalDistortionRngSeed True
EnableInitWeightRngSeed True

EnableHearing True
EnableVoice True
NumSoundFrequencies 2

FightMultiplier 0.0
MinAgentMaxEnergy 1000.0
MaxAgentMaxEnergy 1000.1
EnergyUseMultiplier 0.001

Domains [
    {
      InitAgentsCenterX 0.5
      InitAgentsCenterZ 0.5
      FoodPatches [
      ]
    }
  ]
"""
