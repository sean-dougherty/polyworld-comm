CC=mpigxx

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
		tools\
		utils

CPP_FILES=$(foreach dir, ${SRCDIRS}, $(wildcard ${dir}/*.cp))
CPP_OBJS=$(patsubst %.cp, .bldgnu/cpp/%.o, ${CPP_FILES})

OBJS=${CPP_OBJS}

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

./Polyworld.gnu: ${OBJS}
	echo hi

.bldgnu/cpp/%.o: %.cp
	@mkdir -p $(dir $@)
	${CC} -c -std=c++11 -o $@ ${FLAGS_INCLUDES} $<
