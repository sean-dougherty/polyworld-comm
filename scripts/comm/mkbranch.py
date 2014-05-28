import math

worldsize = 200.0

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
	
	for i in range(len(coords)):
		nc = list(coords[i])
		nc[0] *= -1
		nc[2] *= -1
		coords.append(nc)

	def rotate(x, y):
		theta = math.atan2(y, x)
		length = math.sqrt(x*x + y*y)
		
		return length * math.cos(theta + rotation), length * math.sin(theta + rotation)

	for c in coords:
		c[0], c[1] = rotate(c[0], c[1])
		c[2], c[3] = rotate(c[2], c[3])

	return coords

coords = make_branch(0) + make_branch(0.94) + make_branch(-0.94) + make_branch(2.20) + make_branch(-2.20)

print """\
@version 2

WorldSize %f
MinAgents 200
MaxAgents MinAgents
InitAgents MinAgents
SeedAgents MinAgents

MaxSteps 1000

Barriers [""" % worldsize

for c in coords:
	for i in range(4):
		c[i] /= worldsize

	c[0] += 0.5
	c[2] += 0.5
	c[1] += 0.5
	c[3] += 0.5
	c[1] *= -1
	c[3] *= -1

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

BarrierHeight  1.0

MaxAgentSize 0.75
"""
