#decomposition/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

noinst_LTLIBRARIES	= libdecomposition.la

pkginclude_HEADERS = include/Decomposition_Base.h \
                     include/N_Qubit_Decomposition_Base.h \
                     include/N_Qubit_Decomposition.h \
                     include/Sub_Matrix_Decomposition.h \
                     include/N_Qubit_Decomposition_Cost_Function.h \
                     include/Sub_Matrix_Decomposition_Cost_Function.h \
                     include/Functor_Cost_Function_Gradient.h \
                     include/Functor_Get_Operation_Products.h

libdecomposition_la_SOURCES	= Decomposition_Base.cpp \
                                  Functor_Cost_Function_Gradient.cpp \
                                  N_Qubit_Decomposition_Base.cpp \
                                  N_Qubit_Decomposition_Cost_Function.cpp \
                                  Sub_Matrix_Decomposition.cpp \
                                  Sub_Matrix_Decomposition_Cost_Function.cpp \
                                  Functor_Get_Operation_Products.cpp


AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)


