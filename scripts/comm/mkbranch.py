#!/usr/bin/python

import math
import random
import sys

random_single_food = False
seed_from_run = False
agent_start = None
food_loc = None
food_difficulty = -1
max_steps = 1000

args = sys.argv[1:]
while True:
	if len(args) and args[0] == '--random-single-food':
		random_single_food = True
		seed_random_single_food = int(args[1])
		args = args[2:]
	elif len(args) and args[0] == '--seed-from-run':
		seed_from_run = True
		args = args[1:]
	elif len(args) and args[0] == '--agent-start':
		agent_start = map(float, args[1:3])
		args = args[3:]
	elif len(args) and args[0] == '--food-loc':
		food_loc = map(float, args[1:3])
		args = args[3:]
	elif len(args) and args[0] == '--food-difficulty':
		food_difficulty = int(args[1])
		args = args[2:]
	elif len(args) and args[0] == '--max-steps':
		max_steps = int(args[1])
		args = args[2:]
	elif len(args) and args[0][:2] == '--':
		print 'invalid option:', args[0]
		exit(1)
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
		[-1.5, 2.6, -1.5, 33.61],
		[-1.5, 33.61, -12.25, 44.36],
		[-12.25, 44.36, -20.65, 44.36],
		[-20.65, 44.36, -20.65, 46.67],
		[-20.65, 46.67, -14.56, 46.67],
		[-14.56, 46.67, -21.58, 53.69],
		[-21.58, 53.69, -19.22, 56.04],
		[-19.22, 56.04, -11.97, 48.79],
		[-11.97, 48.79, -11.97, 55.26],
		[-11.97, 55.26, -9.66, 55.26],
		[-9.66, 55.26, -9.66, 46.48],
		[-9.66, 46.48, -1.5, 38.32],
		[-1.5, 38.32, -1.5, 65.38],
		[-1.5, 65.38, -7.79, 71.67],
		[-7.79, 71.67, -6.16, 73.31],
		[-6.16, 73.31, -1.5, 68.65],
		[-1.5, 68.65, -1.5, 81.08],
		[-1.5, 81.08, 0.0, 81.08],
	]

	coords += yreflect(coords)
	
	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)
		c[2], c[3] = rotate(c[2], c[3], rotation)

	return coords

def make_food_patches(rotation, difficulty):
	first_sound = {
		0.0: 2,
		1.05: 1,
		-1.05: 3,
		2.10: -1,
		-2.10: -1
		}

	first_sound = first_sound[rotation]


	def filter_difficulty(dc):
		if difficulty != -1:
			dc = filter(lambda x: x[0] <= difficulty, dc)
		return map(lambda x: x[1], dc)

	coords = filter_difficulty([
			(1, [-9.68, 44.14]),  # 2 1
			(2, [-19.31, 45.47]), # 2 1 1
			(2, [-19.49, 53.93]), # 2 1 2
			(2, [-10.82, 54.03]), # 2 1 3
			(2, [-6.09, 71.58])   # 2 2 1
			])
	coords += yreflect(coords)

	sounds = filter_difficulty([
		(1, [2, 1]),
		(2, [2, 1, 1]),
		(2, [2, 1, 2]),
		(2, [2, 1, 3]),
		(2, [2, 2, 1])
		])
	sounds += filter_difficulty([
		(1, [2, 3]),
		(2, [2, 3, 3]),
		(2, [2, 3, 2]),
		(2, [2, 3, 1]),
		(2, [2, 2, 3])
		])

	coords += filter_difficulty([
		(0, [0.0, 31.05]), # 2
		(1, [0.0, 63.59]), # 2 2
		(2, [0.0, 79.89])  # 2 2 2
	])

	sounds += filter_difficulty([
		(0, [2]),
		(1, [2, 2]),
		(2, [2, 2, 2])
		])

	for i in range(len(sounds)):
		sounds[i][0] = first_sound

	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)

	assert len(coords) == len(sounds)

	return coords, sounds

def make_trunk():
	coords = [
		[-1.5, -2.6, 1.5, -2.6],
	]
	return coords

def make_nest():
	coords = [
		[-1.5, -134.46, -24.30, -148.88],
		[-24.30, -148.88, -24.30, -172.74],
		[-24.30, -172.74, 0.0, -172.74]
	]
	return coords + yreflect(coords)

def make_branch_dist(rotation):
	coords = [
		[0.0, 34.81, -10.81, 45.54],
		[-10.81, 45.54, -21.32, 45.52],
		[-10.81, 45.54, -21.04, 55.48],
		[-10.81, 45.54, -10.82, 56.02],
		[0.0, 65.45, -7.98, 73.61]
	]
	coords += yreflect(coords)

	coords += [
		[0.0, 0.0, 0.0, 34.81],
		[0.0, 34.81, 0.0, 65.45],
		[0.0, 65.45, 0.0, 81.70]
	]

	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)
		c[2], c[3] = rotate(c[2], c[3], rotation)

	return coords

def make_nest_dist():
	coords = [
		[0.0, 0.0, 0.0, -174.46]
	]
	return coords

def make_trunk_bricks():
	coords = [
		[0.0, 0.0]
	]
	colors = [
		[0.6, 0.0, 0.0]
	]

	return coords, colors

def make_branch_bricks(rotation):
	coords = [
		[0.0, 34.81],
		[0.0, 65.45]
	]
	colors = [
		[0.8, 0.0, 0.0],
		[1.0, 0.0, 0.0]
	]

	for c in coords:
		c[0], c[1] = rotate(c[0], c[1], rotation)

	return coords, colors
	

coords = []
food_coords = []
sounds = []
dist_coords = []
brick_coords = []
brick_colors = []

for rotation in [0, 1.05, -1.05, 2.10, -2.10]:
	coords += make_branch(rotation)
	dist_coords += make_branch_dist(rotation)
	brick_coords_, brick_colors_ = make_branch_bricks(rotation)
	brick_coords += brick_coords_
	brick_colors += brick_colors_

coords += make_trunk()
dist_coords += make_nest_dist()
brick_coords_, brick_colors_ = make_trunk_bricks()
brick_coords += brick_coords_
brick_colors += brick_colors_


if food_loc:
	food_coords = [food_loc]
else:
	for rotation in [1.05, -1.05]:
		_food_coords, _sounds = make_food_patches(rotation, food_difficulty)
		food_coords += _food_coords
		sounds += _sounds

if random_single_food:
	random.seed(seed_random_single_food)
	food_index = random.randint(0, len(food_coords)-1)
	food_coords = [food_coords[food_index]]
	sounds = [sounds[food_index]]


if agent_start:
	agent_start = to_polyworld_coords([agent_start])[0]
	nest_centerX = agent_start[0]
	nest_centerY = 1.0 + agent_start[1]
else:
	nest_centerX = 0.5
	nest_centerY = 0.5

nest_sizeX = 1.0 / worldsize
nest_sizeY = 1.0 / worldsize

coords = to_polyworld_coords(coords)
food_coords = to_polyworld_coords(food_coords)
dist_coords = to_polyworld_coords(dist_coords)
brick_coords = to_polyworld_coords(brick_coords)


print """\
@version 2

WorldSize %f
MinAgents 1
MaxAgents 300
InitAgents 80
SeedAgents 80
SeedMutationProbability 0.5
MateWait 0
AgentsAreFood False

MaxSteps %d

SeedGenomeFromRun %s""" % (worldsize, max_steps, seed_from_run)

print "Barriers ["
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
print "]"

print "EnableHearing True"
print "NumSoundFrequencies 3"

if len(sounds) == 1:
	sounds = map(lambda x: x - 1, sounds[0])
	sequence = ','.join(map(str, sounds))

	print """\
SoundPatches [
  {
    SizeX %f
    SizeZ %f
    CenterX %f
    CenterZ %f
    Sequence [%s]
  }
]
""" % (1.0, 1.0, 0.5, 0.5, sequence)

print "DistancePaths ["
for i in range(len(dist_coords)):
	c = dist_coords[i]
	print '  {'
	
	print '    X1', c[0]
	print '    Z1', c[1]
	print '    X2', c[2]
	print '    Z2', c[3]

	print '  }'
	if i != (len(dist_coords) - 1):
		print '  ,'
print "]"

print """
MinFood %d
MaxFood %d
InitFood %d
""" % (len(food_coords),len(food_coords),len(food_coords))

print "SolidBricks False"
print "BrickHeight 0.23"

print """\
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
""" % (nest_centerX, nest_centerY, nest_sizeX, nest_sizeY)

print """
      BrickPatches [
"""
for i in range(len(brick_coords)):
	print """\
        {
          CenterX                   %f
          CenterZ                   %f
          SizeX                     0.001
          SizeZ                     0.001
          BrickCount                1
          BrickColor { R %f; G %f; B %f }
        }
""" % (brick_coords[i][0], 1+brick_coords[i][1], brick_colors[i][0], brick_colors[i][1], brick_colors[i][2])

	if i != (len(brick_coords) - 1):
		print '        ,'

print """\
      ]
"""

foodFraction = 1.0 / len(food_coords)

print """
      FoodPatches [
"""
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
"""

print """\
BarrierHeight  1.0

MaxAgentSize 0.75
RecordGenomes Fittest
RecordPosition Approximate
RecordBrainBehaviorNeurons True
EndOnEat True
FitnessMode MazeFood
YawInit 0
YawEncoding Oppose
EnableYawOpposeThreshold True
MinYawOpposeThreshold 0.0
MaxYawOpposeThreshold 0.1
FightMultiplier 0.0
BodyRedChannel 0.0
MinAgentMaxEnergy 1000.0
MaxAgentMaxEnergy 1000.1
EnergyUseMultiplier 0.001
StickyBarriers True
EnableSpeedFeedback True
EyeHeight 1.2
EnableVisionPitch True
MinVisionPitch -90
MaxVisionPitch -3.4
MinLifeSpan 1000
MaxLifeSpan 1001
EnableTopologicalDistortionRngSeed True
EnableInitWeightRngSeed True
MinMutationRate 0.001
MaxMutationRate 0.005
"""
