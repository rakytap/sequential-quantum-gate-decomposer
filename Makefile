include commonvars.in


all: help inform compilecomomon compileoperations compiledecomposition compiletest

help:
	@echo "To build Quantum Gate Decomposer package:"
	@echo "  make [debug]"
	@echo "       [THREAD=<name> LINKING=<name>]"
	@echo
	@echo "To get help just run make or:"
	@echo "  make help"
	@echo
	@echo "To clean results:"
	@echo "  make clean"
	@echo
	@echo "Main options:"
	@echo "  debug use debug build"
	@echo
	@echo "  THREAD=<name> selects the threding of the result:"
	@echo "      SEQ - sequential version"
	@echo "      PAR - multithreaded version (DEFAULT)"
	@echo "      MIC - multithreaded version for intel MIC devices"
	@echo
	@echo "  LINKING=<name> selects threading of MKL:"
	@echo "      dynamical - dynamical linking (DEFAULT)"
	@echo "      static - static linking for portable version"
	@echo


DEBUG = nodebug

debug: DEBUG=debug
debug: all

inform:
	@echo "------------------------"
	@echo "Bulding Quantum Gate Decomposer package"
	@echo "------------------------"
	@echo



compiletest:
	@$(MAKE) $(DEBUG) -C ./test/

compilelbfgs:
	@$(MAKE) $(DEBUG) -C ./lbfgs/

compilecomomon:
	@$(MAKE) $(DEBUG) -C ./common/

compileoperations:
	@$(MAKE) $(DEBUG) -C ./operations/

compiledecomposition:
	@$(MAKE) $(DEBUG) -C ./decomposition/

compilemex:
	@$(MAKE) $(DEBUG) -C ./source/

buildinglib:
	@$(MAKE) $(DEBUG) -C ./lib/

clean: 
	@$(MAKE) clean -C ./source/
