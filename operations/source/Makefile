# Makefile MEX shared libraries
include ../../commonvars.in

OBJECTDIR = $(QGDDIR)/operations/bin/$(PLATFORM)

TARGETS = Operation \
          U3

SOURCES := $(patsubst %.cxx,%,$(wildcard *.cxx) ) 



# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -c -fopenmp -I$(QGDDIR)/operations/include -I$(QGDDIR)/common/include -I$(MKLROOT)/include  -g -O2 -fPIC

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
	ar rc $(OBJECTDIR)/liboperations.a $(OBJECTDIR)/*.o



.PHONY: compile $(TARGETS)


createObjects : $(TARGETS)

$(TARGETS) :
	@echo
	@echo "--------------------"
	@echo "Creating executable: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) ./$@.cpp -o $(OBJECTDIR)/$@.o ;\

	

