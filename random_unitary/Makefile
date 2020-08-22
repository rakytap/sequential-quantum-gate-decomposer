include ../commonvars.in

DEBUG = nodebug

INCDIRS = source


all: compileSubdirs $(INCDIRS)

# Set the DEBUG flag
debug: DEBUG=debug
debug: all
nodebug: all


.PHONY: compile $(INCDIRS)


compileSubdirs : $(INCDIRS)

$(INCDIRS) :
	@echo "Compile "$@
	@$(MAKE) $(DEBUG) -C ./$@ 


