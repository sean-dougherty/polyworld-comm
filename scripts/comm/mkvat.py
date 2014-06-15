#!/usr/bin/python

import sys

Low_Mutation_Rate = 0.0005
Med_Mutation_Rate = 0.001
High_Mutation_Rate = 0.001
#High_Mutation_Rate = 0.05

seed_from_run = False

mutation_rate = Low_Mutation_Rate

args = sys.argv[1:]
while True:
	if len(args) and args[0] == '--seed-from-run':
		seed_from_run = True
		args = args[1:]
	elif len(args) and args[0] == '--high-mutation':
		mutation_rate = High_Mutation_Rate
		args = args[1:]
	elif len(args) and args[0] == '--med-mutation':
		mutation_rate = Med_Mutation_Rate
		args = args[1:]
	elif len(args) and args[0] == '--low-mutation':
		mutation_rate = Low_Mutation_Rate
		args = args[1:]
	elif len(args) and args[0][:2] == '--':
		print 'invalid option:', args[0]
		exit(1)
	else:
		break


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
SeedGenomeFromRun %s
""" % seed_from_run

print """\
Vision False

InitSeed 1

MinAgents 1
MaxAgents 150
InitAgents 40
SeedAgents 40
NumberFittest 10
MinLifeSpan 100000
MaxLifeSpan 100001
SeedMutationProbability 0.5
MinCrossoverPoints 1
MaxCrossoverPoints 4
MateWait 0

EnableTopologicalDistortionRngSeed True
EnableInitWeightRngSeed True

EnableHearing True
EnableVoice True
NumSoundFrequencies 2

RecordBrainBehaviorNeurons True

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
