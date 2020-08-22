# Makefile
include ../../commonvars.in

OBJECTDIR = $(QGDDIR)/common/bin/$(PLATFORM)

TARGETS = common

SOURCES := $(patsubst %.cxx,%,$(wildcard *.cxx) ) 


# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -c -fopenmp -I$(QGDDIR)/common/include -I$(MKLROOT)/include -fPIC -g -O2 

ifeq ($(THREAD),MIC)
  FLAGS_COMPILE += -DMIC -mmic
endif

all: cleanup createoutputdir createObjects $(TARGETS) createstatic

# Set the DEBUG flag
debug: FLAGS_COMPILE += -DDEBUG
debug: all

nodebug: all

cleanup :
	rm -rf $(OBJECTDIR)/*

createoutputdir:
	mkdir -p $(OBJECTDIR)

createstatic :
	ar rc $(OBJECTDIR)/libcommon.a $(OBJECTDIR)/*.o



.PHONY: compile $(TARGETS)


createObjects : $(TARGETS)

$(TARGETS) :
	@echo
	@echo "--------------------"
	@echo "Creating library: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) ./$@.cpp -o $(OBJECTDIR)/$@.o ;\

	

