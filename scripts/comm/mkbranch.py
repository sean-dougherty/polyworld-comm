#!/usr/bin/python

import math
import random
import sys

random_single_food = False
seed_from_run = False

args = sys.argv[1:]
while True:
	if len(args) and args[0] == '--random-single-food':
		random_single_food = True
		seed_random_single_food = int(args[1])
		args = args[2:]
	elif len(args) and args[0] == '--seed-from-run':
		seed_from_run = True
		args = args[1:]
	else:
		break

worldsize = 375.0

def yreflect(coords):
	result = []
	for orig in coords:
		reflected = list(orig)
		reflected[0] *= -1
		if len(reflected) > 2:
			reflected[2] *= -1
		result.append(reflected)

	return result

def xtranslate(coords, shift):
	result = []
	for orig in coords:
		trans = list(orig)
		trans[0] += shift
		if len(orig) > 2:
			trans[2] += shift
		result.append(trans)

	return result

def ytranslate(coords, shift):
	result = []
	for orig in coords:
		trans = list(orig)
		trans[1] += shift
		if len(orig) > 2:
			trans[3] += shift
		result.append(trans)

	return result

def rotate(x, y, rotation):
	theta = math.atan2(y, x)
	length = math.sqrt(x*x + y*y)
	
	return length * math.cos(theta + rotation), length * math.sin(theta + rotation)

def to_polyworld_coords(coords):
	result = []
	for c in coords:
		rc = list(c)
		for i in range(len(c)):
			rc[i] /= worldsize
			rc[i] += 0.5
		
		rc[1] *= -1
		if len(c) > 2:
			rc[3] *= -1

		result.append(rc)

	return result

def make_branch(rotation):
	coords = [
		[-1.48, 0.13, -1.48, 31.14],
		[-1.48, 31.14, -12.23, 41.89],
		[-12.23, 41.89, -20.63, 41.89],
		[-20.63, 41.89, -20.63, 44.20],
		[-20.63, 44.20, -14.54, 44.20],
		[-14.54, 44.20, -21.56, 51.22],
		[-21.56, 51.22, -19.20, 53.57],
		[-19.20, 53.57, -11.95, 46.32],
		[-11.95, 46.32, -11.95, 52.79],
		[-11.95, 52.79, -9.64, 52.79],
		[-9.64, 52.79, -9.64, 44.01],
		[-9.64, 44.01, -1.48, 35.85],
		[-1.48, 35.85, -1.48, 62.91],
		[-1.48, 62.91, -7.77, 69.20],
		[-7.77, 69.20, -6.14, 70.84],
		[-6.14, 70.84, -1.48, 66.18],
		[-1.48, 66.18, -1.48, 78.61],
		[-1.48, 78.61, 0.00, 78.61]
	]

	coords = xtranslate(coords, -0.02)
	coords = ytranslate(coords, 2.47)
	coords += yreflect(coords)
	
	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)
		c[2], c[3] = rotate(c[2], c[3], rotation)

	return coords

def make_food_patches(rotation):
	axis_coords = [
		[0, 31.14],
		[0, 60.81],
		[0, 77.45]
	]
	axis_coords = xtranslate(axis_coords, -0.02)
	axis_coords = ytranslate(axis_coords, 2.47)

	coords = [
		[-10.29, 42.0],
		[-20.0, 43.0],
		[-20.0, 51.7],
		[-11.0, 52.0],
		[-6.42, 69.42],
	]
	coords += yreflect(coords)
	coords = xtranslate(coords, -0.02)
	coords = ytranslate(coords, 2.47)

	coords = axis_coords + coords
	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)

	return coords

def make_trunk():
	coords = [
		[-2.05, -137.93, -1.5, -2.56]
	]
	coords = ytranslate(coords, 2.47)
	coords = coords + yreflect(coords)
	return coords

def make_nest():
	coords = [
		[-1.75, -136.75, -24.85, -151.35],
		[-24.85, -151.35, -24.28, -175.21],
		[-24.28, -175.21, 0, -175.21]
	]
	coords = ytranslate(coords, 2.47)
	return coords + yreflect(coords)

coords = []
food_coords = []

for rotation in [0, 1.05, -1.05, 2.10, -2.10]:
	coords += make_branch(rotation)
	food_coords += make_food_patches(rotation)

coords += make_trunk() + make_nest()

if random_single_food:
	random.seed(seed_random_single_food)
	food_index = random.randint(0, len(food_coords)-1)
	food_coords = [food_coords[food_index]]


nest_walls = make_nest()
nest_centerX = 0.5
nest_centerY = 0.5 - ((nest_walls[1][1] + nest_walls[1][3]) / 2.0 / worldsize)
#nest_sizeX = 0.75 * ((abs(nest_walls[1][0]) * 2) / worldsize)
#nest_sizeY = 0.75 * (abs(nest_walls[1][1] - nest_walls[1][3]) / worldsize)
nest_sizeX = 1.0 / worldsize
nest_sizeY = 1.0 / worldsize

coords = to_polyworld_coords(coords)
food_coords = to_polyworld_coords(food_coords)


print """\
@version 2

WorldSize %f
MinAgents 1
MaxAgents 300
InitAgents 200
SeedAgents 200

MaxSteps 1000

SeedGenomeFromRun %s

Barriers [""" % (worldsize, seed_from_run)


for i in range(len(coords)):
	c = coords[i]
	print '  {'
	
	print '    X1', c[0]
	print '    Z1', c[1]
	print '    X2', c[2]
	print '    Z2', c[3]

	print '  }'
	if i != (len(coords) - 1):
		print '  ,'


print """\
]

Domains [
    {
      CenterX                   0.5
      CenterZ                   0.5
      SizeX                     1.0
      SizeZ                     1.0
      InitAgentsCenterX			%f
      InitAgentsCenterZ			%f
      InitAgentsSizeX		    %f
      InitAgentsSizeZ			%f
      FoodPatches [
""" % (nest_centerX, nest_centerY, nest_sizeX, nest_sizeY)

foodFraction = 1.0 / len(food_coords)

for i in range(len(food_coords)):
	print """\
        {
          FoodFraction              %f
          MaxFoodFraction           1.0
          MaxFoodGrownFraction      1.0
          CenterX                   %f
          CenterZ                   %f
          SizeX                     0.001
          SizeZ                     0.001
        }
""" % (foodFraction, food_coords[i][0], 1+food_coords[i][1])

	if i != (len(food_coords) - 1):
		print '        ,'

print """\
      ]
    }
  ]

BarrierHeight  1.0

MaxAgentSize 0.75
RecordGenomes Fittest
RecordPosition Approximate
RecordBrainBehaviorNeurons True
EndOnEat True
FitnessMode MazeFood
YawInit 0
YawOpposeThreshold 0.1
FightMultiplier 0.0
MinAgentMaxEnergy 1000.0
MaxAgentMaxEnergy 1000.1
EnergyUseMultiplier 0.05
StickyBarriers True
EnableSpeedFeedback True
EyeHeight 1.2
EnableVisionPitch True
MinVisionPitch -90
MaxVisionPitch -3.4
MinLifeSpan 1000
MaxLifeSpan 1001
"""
