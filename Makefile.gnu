CC=mpigxx
NVCC=nvcc --compiler-bindir /usr/bin

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

SYSTEM_INCLUDES= \
/usr/include \
/share/apps/cuda/cuda6/extras/CUPTI/include \
/share/apps/cuda/cuda6/extras/CUPTI/include/GL \
/share/apps/cuda/cuda6/include \
/share/apps/gsl/include \
/share/apps/python/2.7.5/include/python2.7


INCLUDES=${SRCDIRS} ${SYSTEM_INCLUDES}
FLAGS_INCLUDES=$(foreach dir, ${INCLUDES}, -I${dir})


LIBS=-lz -lgsl -lgslblas -lgomp

.PHONY: clean

./Polyworld.gnu: ${OBJS}
	@echo ${DEPENDS}

clean:
	rm -rf .bldgnu

.bldgnu/obj/cpp/%.o: %.cp
	@mkdir -p $(dir $@)
	${CC} -MMD -c -std=c++11 -o $@ ${FLAGS_INCLUDES} $<

.bldgnu/obj/cuda/%.o: %.cu
	@mkdir -p $(dir $@)
	${NVCC} -c -arch=sm_13 -o $@ ${FLAGS_INCLUDES} $<

-include ${DEPENDS}
