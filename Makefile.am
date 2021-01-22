AUTOMAKE_OPTIONS = foreign subdir-objects

ACLOCAL_AMFLAGS = -I m4
SUBDIRS = qgd common operations decomposition random_unitary test_standalone

SUBLIBS = common/libcommon.la operations/liboperations.la decomposition/libdecomposition.la random_unitary/librandomunitary.la

EXTRA_DIST = common/include/common.h \
             common/include/matrix.h \
             common/include/matrix_base.hpp \
             common/include/QGDTypes.h \
             common/include/dot.h \
             operations/include/Operation.h \
             operations/include/CNOT.h \
             operations/include/U3.h \
             operations/include/Operation_block.h \
             decomposition/include/Decomposition_Base.h \
             decomposition/include/N_Qubit_Decomposition_Base.h \
             decomposition/include/N_Qubit_Decomposition.h \
             decomposition/include/Sub_Matrix_Decomposition.h \
             decomposition/include/Functor_Cost_Function_Gradient.h \
             decomposition/include/Functor_Get_Operation_Products.h \
             decomposition/include/N_Qubit_Decomposition_Cost_Function.h \
             decomposition/include/Sub_Matrix_Decomposition_Cost_Function.h \
             random_unitary/include/Random_Unitary.h \
             README.md \
             example.py \
             qgd_python/N_Qubit_Decomposition.py \
             qgd_python/__init__.py \
             Umtx.mat \
             LICENSE \
             Doxygen

lib_LTLIBRARIES = libqgd.la
pkginclude_HEADERS = python_interface.h
libqgd_la_SOURCES = python_interface.cpp
AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)
libqgd_la_LIBADD = $(GSL_LIBADD) $(SUBLIBS)

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 


install: install-recursive
	cp -r qgd_python $(prefix)/
	cp example.py $(prefix)/
	cp Umtx.mat $(prefix)/
	cp LICENSE $(prefix)/


doxygen: 
	cd Doxygen && doxygen Doxyfile && cd ..

doxygen-install:
	mkdir -p $(prefix)/Docs && cp -r Docs/html $(prefix)/Docs
      
	
