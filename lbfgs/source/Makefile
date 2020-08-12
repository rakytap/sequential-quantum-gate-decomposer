# Makefile MEX shared libraries
include ../../commonvars.in

OBJECTDIR = $(QGDDIR)/lbfgs/bin/$(PLATFORM)

TARGETS = lbfgs

SOURCES := $(patsubst %.c,%,$(wildcard *.c) ) 



# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -msse2 -DUSE_SSE -c -fopenmp -I$(QGDDIR)/lbfgs/include  -g -O2 -fPIC

ifeq ($(THREAD),MIC)
  FLAGS_COMPILE += -DMIC -offload-attribute-target=mic -offload-option,mic,compiler," -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5"
endif

all: cleanup createoutputdir createObjects $(TARGETS) createstatic

# Set the DEBUG flag
debug: FLAGS_COMPILE += -DDEBUG
debug: OUTPUTDIR = $(QGDDIR)/bin/DEBUG
debug: all

nodebug: OUTPUTDIR = $(QGDDIR)/bin
nodebug: all

cleanup :
	rm -rf $(OBJECTDIR)/*

createoutputdir:
	mkdir -p $(OBJECTDIR)


createstatic :
	ar rc $(OBJECTDIR)/liblbfgs.a $(OBJECTDIR)/*.o



.PHONY: compile $(TARGETS)


createObjects : $(TARGETS)

$(TARGETS) :
	@echo
	@echo "--------------------"
	@echo "Creating library: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) ./$@.c -o $(OBJECTDIR)/$@.o ;\

	

