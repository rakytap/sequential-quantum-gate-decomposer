#operations/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

noinst_LTLIBRARIES	= liboperations.la

pkginclude_HEADERS = include/Operation.h include/CNOT.h include/U3.h include/Operation_block.h

liboperations_la_SOURCES  = Operation.cpp CNOT.cpp U3.cpp Operation_block.cpp

AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)
