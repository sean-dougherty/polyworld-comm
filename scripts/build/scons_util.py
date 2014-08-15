import inspect
import os
import re
import sys


################################################################################
###
### FUNCTION init_env
###
################################################################################
def init_env(env):
	env.Append( LIBPATH = ['/usr/lib64',
						   os.path.expanduser('~/lib')] )

	if PFM == 'linux':
		env.Append( DEFAULTFRAMEWORKPATH = [] )
		env.Append( FRAMEWORKPATH = [] )
		env.Append( LIBPATH = ['/usr/lib/x86_64-linux-gnu'] )
	elif PFM == 'mac':
		env.Append( DEFAULTFRAMEWORKPATH = ['/System/Library/Frameworks',
											'/Library/Frameworks'] )
		env.Append( FRAMEWORKPATH = [] )

	return env

################################################################################
###
### FUNCTION import_cuda
###
################################################################################
def import_cuda(env):
	thisFile = inspect.getabsfile(import_cuda)
	thisDir = os.path.dirname(thisFile)
	env['CUDA_TOOLKIT_PATH'] = '/share/apps/cuda/cuda6'
	env['CUDA_SDK_PATH'] = '/share/apps/cuda/cuda6'
	env.Tool('cuda', toolpath = [os.path.join(thisDir)])	

################################################################################
###
### FUNCTION import_OpenGL
###
################################################################################
def import_OpenGL(env):
	inc = which('gl.h',
				dir = True,
				path = ['/share/apps/cuda/cuda6/extras/CUPTI/include/GL',
						'/usr/include/GL',
						'/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers',
						'/System/Library/Frameworks/AGL.framework/Versions/A/Headers/'],
				err = 'Cannot locate OpenGL')
	incs = [inc, os.path.join(inc, os.path.dirname(inc))]
	env.Append( CPPPATH = incs )

	if PFM == 'linux':
		addlib(env, 'GL')
		addlib(env, 'GLU')
	elif PFM == 'mac':
		addlib(env, 'AGL')
		addlib(env, 'OpenGL')

################################################################################
###
### FUNCTION import_OpenMP
###
################################################################################
def import_OpenMP(env):
	env.Append( LIBS = 'gomp' )
	env.Append( CPPFLAGS = '-fopenmp' )

################################################################################
###
### FUNCTION import_GSL
###
################################################################################
def import_GSL(env):
	env.Append( CPPPATH = ['/share/apps/gsl/include'] ) # usc cluster
	env.Append( LIBPATH = ['/share/apps/gsl/lib'] ) # usc cluster
	
	
	addlib(env, 'gsl')
	addlib(env, 'gslcblas')

################################################################################
###
### FUNCTION import_zlib
###
################################################################################
def import_zlib(env):
	addlib(env, 'z')

################################################################################
###
### FUNCTION import_pthread
###
################################################################################
def import_pthread(env):
	addlib(env, 'pthread')

################################################################################
###
### FUNCTION import_rt
###
################################################################################
def import_rt(env):
	addlib(env, 'rt')

################################################################################
###
### FUNCTION import_dlsym
###
################################################################################
def import_dlsym(env):
	addlib(env, 'dl')

################################################################################
###
### FUNCTION import_python
###
################################################################################
def import_python(env, version=None):
	if PFM == 'linux':
		if not version: version='2.7'
		#env.Append( CPPPATH = ['/usr/include/python'+version, '/usr/local/include/python'+version] )
		#env.Append( LIBPATH = ['/share/apps/python/2.7.5/lib/'] )
		env.Append( CPPPATH = ['/share/apps/python/2.7.5/include/python2.7'] )
		env.Append( LIBPATH = ['/share/apps/python/2.7.5/lib'] )
	elif PFM == 'mac':
		if not version: version='2.7'
		env.Append( CPPPATH = ['/System/Library/Frameworks/Python.framework/Versions/'+version+'/include/python'+version] )
	else:
		assert(False)

	addlib(env, 'python'+version)

################################################################################
###
### FUNCTION export_dynamic
###
################################################################################
def export_dynamic( env ):
	if PFM == 'linux':
		env.Append( LINKFLAGS = ['-rdynamic'] )
	elif PFM == 'mac':
		env.Append( LINKFLAGS = ['-undefined', 'dynamic_lookup'] )
	else:
		assert( False )

################################################################################
###
### FUNCTION find
###
################################################################################
def find(topdir = '.',
		 type = None,
		 name = None,
		 regex = None,
		 ignore = ['CVS'] ):

	assert((name == None and regex == None)
		   or (name != None) ^ (regex != None))
	if type:
		assert(type in ['file', 'dir'])

	result = []

	if(name):
		# convert glob syntax to regex
		pattern = name.replace('.','\\.').replace('*','.*') + '$'
	elif(regex == None):
		pattern = '.*'
	else:
		pattern = regex

	cregex = re.compile(pattern)

	for dirinfo in os.walk(topdir):
		dirname = dirinfo[0]
		if type == None:
			children = dirinfo[1] + dirinfo[2]
		elif type == 'dir':
			children = dirinfo[1]
		elif type == 'file':
			children = dirinfo[2]
		else:
			assert(false)

		for filename in children:
			if filename in ignore:
				continue

			path = os.path.join(dirname, filename)

			if cregex.match(filename):
			   result.append(path)

			if os.path.isdir(path) and os.path.islink(path):
				result += find(path,
							   type,
							   name,
							   regex)

	return result

################################################################################
###
### FUNCTION which
###
################################################################################
def which(name,
		  dir = False,
		  path = os.environ['PATH'].split(':'),
		  err = None):

	for a_path in path:
		p = os.path.join(a_path, name)
		if os.path.exists(p):
			abspath = os.path.abspath(p)
			if dir:
				return os.path.dirname(abspath)
			else:
				return abspath
	if err:
		print err
		sys.exit(1)
	else:
		return None

################################################################################
###
### FUNCTION exclude
###
################################################################################
def exclude(dirs, exclude):
	def test(dir):
		for x in exclude:
			if dir.startswith(x):
				return False
		return True

	return filter(test, dirs)

################################################################################
###
### FUNCTION relocate_paths
###
################################################################################
def relocate_paths(sources, oldtop, newtop):
	n = len(str(oldtop))

	return map(lambda path: str(newtop) + '/' + path[n:],
			   sources)

################################################################################
###
### FUNCTION libsuffixes
###
################################################################################
def libsuffixes():
	if PFM == 'linux':
		return ['.so', '.a']
	elif PFM == 'mac':
		return ['.dylib', '.so', '.a']
	else:
		assert(false)

################################################################################
###
### FUNCTION libprefix
###
################################################################################
def libprefix():
	if PFM == 'linux' or PFM == 'mac':
		return 'lib'
	else:
		assert(false)

################################################################################
###
### FUNCTION addlib
###
################################################################################
def addlib(env, libname, path = None):
	if not path:
		path = env['LIBPATH'] + env['FRAMEWORKPATH'] + env['DEFAULTFRAMEWORKPATH']

	for dir in path:
		if os.path.exists( os.path.join(dir, libname + '.framework') ):
				env.Append( FRAMEWORKS = [libname] )
				if dir not in env['DEFAULTFRAMEWORKPATH']:
					env.AppendUnique( FRAMEWORKPATH = [dir] )
				return
		else:
			for libsuffix in libsuffixes():
				lib = libprefix() + libname + libsuffix

				if os.path.exists( os.path.join(dir, lib) ):
					env.Append( LIBS = [libname] )
					env.AppendUnique( LIBPATH = [dir])
					return

	print 'Failed locating library', libname
	sys.exit(1)

################################################################################
###
### FUNCTION uname
###
################################################################################
def uname():
	type = os.popen('uname').readlines()[0].strip()

	if type.startswith('Darwin'):
		return 'mac'
	elif type.startswith('Linux'):
		return 'linux'
	else:
		assert(false)

PFM = uname()
