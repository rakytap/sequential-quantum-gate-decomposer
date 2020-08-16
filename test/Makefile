# Makefile MEX shared libraries
include ../commonvars.in

OBJECTDIR = $(QGDDIR)/bin/$(PLATFORM)

TARGETS = decomposition_test

SOURCES := $(patsubst %.cxx,%,$(wildcard *.cxx) ) 


# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -c -fopenmp -I$(GSLINCLUDEDIR) -I$(QGDDIR)/operations/include -I$(QGDDIR)/decomposition/include -I$(QGDDIR)/common/include -I$(MKLROOT)/include  -g -O2 -fPIC
LINKFLAGS = -shared -g -O2 -Wl,--start-group $(GSLLIBDIR)/libgsl.a  $(QGDDIR)/decomposition/bin/$(PLATFORM)/libdecomposition.a $(QGDDIR)/operations/bin/$(PLATFORM)/liboperations.a $(QGDDIR)/common/bin/$(PLATFORM)/libcommon.a -Wl,--end-group -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl  -lmkl_avx -lmkl_def  -lmkl_intel_thread -liomp5

ifeq ($(THREAD),MIC)
  FLAGS_COMPILE += -DMIC -offload-attribute-target=mic -offload-option,mic,compiler," -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5"
endif

all: cleanup createoutputdir createObjects $(TARGETS)

# Set the DEBUG flag
debug: FLAGS_COMPILE += -DDEBUG
debug: OUTPUTDIR = $(QGDDIR)/lib/DEBUG
debug: all

nodebug: OUTPUTDIR = $(QGDDIR)/lib
nodebug: all

cleanup :
	rm -rf $(OBJECTDIR)/*

createoutputdir:
	mkdir -p $(OBJECTDIR)


.PHONY: compile $(TARGETS)


createObjects : $(TARGETS)

$(TARGETS) :
	@echo
	@echo "--------------------"
	@echo "Creating test executable: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) $@.cpp -o $(OBJECTDIR)/$@.o ;\
	$(COMPILER_C) $(OBJECTDIR)/$@.o  -o $(OUTPUTDIR)/$@.so  $(LINKFLAGS)

	


