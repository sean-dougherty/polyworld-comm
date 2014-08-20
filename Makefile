include Makefile.conf

SOURCES_CPP=$(shell find src -name "*.cp")
OBJS_CPP=$(patsubst %.cp, .bld/obj/cpp/%.o, ${SOURCES_CPP})

SOURCES_CUDA=$(shell find src -name "*.cu")
OBJS_CUDA=$(patsubst %.cu, .bld/obj/cuda/%.o, ${SOURCES_CUDA})

OBJS=${OBJS_CPP} ${OBJS_CUDA}

INCLUDES=$(shell find src -type d) ${SYSTEM_INCLUDES}
FLAGS_INCLUDES=$(foreach dir, ${INCLUDES}, -I${dir})

LIBS=z gsl gslcblas gomp cudart GL GLU python2.7
FLAGS_LIBS=$(foreach name, ${LIBS}, -l${name})
FLAGS_LIBS_PATH=$(foreach dir, ${LIBS_PATH}, -L${dir})

.PHONY: default cppprops clean
default: ./Polyworld
cppprops: .bld/cppprops/libcppprops.so

clean:
	rm -rf .bld

#
# Targets
#
./Polyworld: ${OBJS}
	${LD} -rdynamic ${OBJS} ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} -o $@

.bld/cppprops/libcppprops.so: .bld/cppprops/generated.cpp
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
