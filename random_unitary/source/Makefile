# Makefile MEX shared libraries
include ../../commonvars.in

OBJECTDIR = $(QGDDIR)/random_unitary/bin/$(PLATFORM)

TARGETS = Random_Unitary 



# Compiler options for the mkl_QGD library
FLAGS_COMPILE =  -c -fopenmp -I$(GSLINCLUDEDIR) -I$(QGDDIR)/random_unitary/include -I$(QGDDIR)/common/include -I$(MKLROOT)/include  -g -O2 -fPIC

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
	ar rc $(OBJECTDIR)/librandomunitary.a $(OBJECTDIR)/*.o



.PHONY: compile $(TARGETS)


createObjects : $(TARGETS)

$(TARGETS) :
	@echo
	@echo "--------------------"
	@echo "Creating library: "$@
	@echo "--------------------"
	$(COMPILER_C) $(FLAGS_COMPILE) ./$@.cpp -o $(OBJECTDIR)/$@.o ;\

	

