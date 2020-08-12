include ../../commonvars.in

OBJECTDIR = $(QGDDIR)/lbfgs/test/bin/$(PLATFORM)

TARGETS = sample

SOURCES := $(patsubst %.cpp,%,$(wildcard *.cpp) ) 



# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -msse2 -DUSE_SSE -c -I$(QGDDIR)/lbfgs/include   -g -O2 -fPIC
LINKFLAGS = -g -O2 -Wl,--start-group $(QGDDIR)/lbfgs/bin/$(PLATFORM)/liblbfgs.a -Wl,--end-group -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl  -lmkl_avx -lmkl_def  -lmkl_intel_thread -liomp5

ifeq ($(THREAD),MIC)
  FLAGS_COMPILE += -DMIC -offload-attribute-target=mic -offload-option,mic,compiler," -L${MKLROOT}/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5"
endif

all: cleanup createoutputdir createObjects $(TARGETS)

# Set the DEBUG flag
debug: FLAGS_COMPILE += -DDEBUG
debug: OUTPUTDIR = $(QGDDIR)/lbfgs/test
debug: all

nodebug: OUTPUTDIR = $(QGDDIR)/lbfgs/test
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
	@echo "Creating executable: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) ./$@.cpp -o $(OBJECTDIR)/$@.o ;\
	$(COMPILER_C) $(OBJECTDIR)/$@.o  -o $(OUTPUTDIR)/$@  $(LINKFLAGS)

	

