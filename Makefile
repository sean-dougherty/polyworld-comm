include Makefile.conf

SOURCES_CPP=$(shell find src -name "*.cp")
OBJS_CPP=${SOURCES_CPP:%.cp=.bld/obj/cpp/%.o}

SOURCES_CUDA=$(shell find src -name "*.cu")
OBJS_CUDA=${SOURCES_CUDA:%.cu=.bld/obj/cuda/%.o}

OBJS=${OBJS_CPP} ${OBJS_CUDA}

INCLUDES=$(shell find src -type d) ${SYSTEM_INCLUDES}
FLAGS_INCLUDES=${INCLUDES:%=-I%}

LIBS=z gsl gslcblas gomp cudart GL GLU
FLAGS_LIBS=${LIBS:%=-l%}
FLAGS_LIBS_PATH=${LIBS_PATH:%=-L%})

.PHONY: default cppprops clean
default: ./Polyworld
cppprops: .bld/cppprops/libcppprops.so

clean:
	rm -rf .bld

#
# Targets
#

# The main Polyworld executable
./Polyworld: ${OBJS}
	${LD} -rdynamic ${OBJS} ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} -o $@

# A dynamically generated so in the run directory, created from dynamic properties
# in the worldfile.
%/.cppprops/libcppprops.so: %/.cppprops/generated.cpp
	@mkdir -p $(dir $@)
	${CC} -fPIC -o $(dir $@)/generated.o ${FLAGS_INCLUDES} $<
	${LD} -shared -o $@ ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} $(dir $@)/generated.o

#
# Patterns
#
.bld/obj/cpp/%.o: %.cp
	@mkdir -p $(dir $@)
	${CC} -o $@ ${FLAGS_INCLUDES} $<

.bld/obj/cuda/%.d: %.cu
	@mkdir -p $(dir $@)
	${CC_CUDA_DEPENDS} ${FLAGS_INCLUDES} $< > $@.tmp
	@cat $@.tmp | sed 's,.*\.o[[:space:]]*:,$@ :,g' > $@
	@rm $@.tmp

.bld/obj/cuda/%.o: %.cu
	@mkdir -p $(dir $@)
	${CC_CUDA} -o $@ ${FLAGS_INCLUDES} $<

#
# Automatic dependency generation
#
DEPENDS=${OBJS:%.o=%.d}
-include ${DEPENDS}
