#decomposition/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

noinst_LTLIBRARIES	= libdecomposition.la

pkginclude_HEADERS = include/Decomposition_Base.h include/N_Qubit_Decomposition.h include/Sub_Matrix_Decomposition.h

libdecomposition_la_SOURCES	= Decomposition_Base.cpp N_Qubit_Decomposition.cpp Sub_Matrix_Decomposition.cpp

AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)


