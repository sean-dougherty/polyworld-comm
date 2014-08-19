include Makefile.conf

SRCDIRS=agent\
		app\
		brain\
		brain/groups\
		brain/sheets\
		complexity\
		debugger\
		environment\
		genome\
		genome/groups\
		genome/sheets\
		graphics\
		logs\
		main\
		proplib\
		utils

SOURCES_CPP=$(foreach dir, ${SRCDIRS}, $(wildcard ${dir}/*.cp))
OBJS_CPP=$(patsubst %.cp, .bld/obj/cpp/%.o, ${SOURCES_CPP})

SOURCES_CUDA=$(foreach dir, ${SRCDIRS}, $(wildcard ${dir}/*.cu))
OBJS_CUDA=$(patsubst %.cu, .bld/obj/cuda/%.o, ${SOURCES_CUDA})

OBJS=${OBJS_CPP} ${OBJS_CUDA}
DEPENDS=${OBJS:%.o=%.d}

INCLUDES=${SRCDIRS} ${SYSTEM_INCLUDES}
FLAGS_INCLUDES=$(foreach dir, ${INCLUDES}, -I${dir})

LIBS=z gsl gslcblas gomp cudart GL GLU python2.7
FLAGS_LIBS=$(foreach name, ${LIBS}, -l${name})
FLAGS_LIBS_PATH=$(foreach dir, ${LIBS_PATH}, -L${dir})

.PHONY: clean cppprops

default: ./Polyworld

cppprops: .bld/cppprops/libcppprops.so

./Polyworld: ${OBJS}
	${LD} -rdynamic ${OBJS} ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} -o $@

.bld/cppprops/libcppprops.so: .bld/cppprops/generated.cpp
	${CC} -std=c++11 -shared -fPIC -o $@ ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} ${FLAGS_INCLUDES} $<

clean:
	rm -rf .bld

.bld/obj/cpp/%.o: %.cp
	@mkdir -p $(dir $@)
	${CC} -MMD -c -std=c++11 -o $@ ${FLAGS_INCLUDES} $<

.bld/obj/cuda/%.d: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -M ${FLAGS_INCLUDES} $< > $@.tmp
	@cat $@.tmp | sed 's,.*\.o[[:space:]]*:,$@ :,g' > $@
	@rm $@.tmp

.bld/obj/cuda/%.o: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -c -o $@ ${FLAGS_INCLUDES} $<

-include ${DEPENDS}
