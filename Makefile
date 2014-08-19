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
OBJS_CPP=$(patsubst %.cp, .bldgnu/obj/cpp/%.o, ${SOURCES_CPP})

SOURCES_CUDA=$(foreach dir, ${SRCDIRS}, $(wildcard ${dir}/*.cu))
OBJS_CUDA=$(patsubst %.cu, .bldgnu/obj/cuda/%.o, ${SOURCES_CUDA})

OBJS=${OBJS_CPP} ${OBJS_CUDA}
DEPENDS=${OBJS:%.o=%.d}

INCLUDES=${SRCDIRS} ${SYSTEM_INCLUDES}
FLAGS_INCLUDES=$(foreach dir, ${INCLUDES}, -I${dir})

LIBS=z gsl gslcblas gomp cudart GL GLU python2.7
FLAGS_LIBS=$(foreach name, ${LIBS}, -l${name})
FLAGS_LIBS_PATH=$(foreach dir, ${LIBS_PATH}, -L${dir})

.PHONY: clean

default: ./Polyworld

./Polyworld: ${OBJS}
	${LD} ${OBJS} ${FLAGS_LIBS} ${FLAGS_LIBS_PATH} -o $@

clean:
	rm -rf .bldgnu

.bldgnu/obj/cpp/%.o: %.cp
	@mkdir -p $(dir $@)
	${CC} -MMD -c -std=c++11 -o $@ ${FLAGS_INCLUDES} $<

.bldgnu/obj/cuda/%.d: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -M ${FLAGS_INCLUDES} $< > $@.tmp
	@cat $@.tmp | sed 's,.*\.o[[:space:]]*:,$@ :,g' > $@
	@rm $@.tmp

.bldgnu/obj/cuda/%.o: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -c -o $@ ${FLAGS_INCLUDES} $<

-include ${DEPENDS}
