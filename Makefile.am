AUTOMAKE_OPTIONS = foreign subdir-objects

ACLOCAL_AMFLAGS = -I m4
SUBDIRS = qgd common operations decomposition random_unitary test_standalone

SUBLIBS = common/libcommon.la operations/liboperations.la decomposition/libdecomposition.la random_unitary/librandomunitary.la

EXTRA_DIST = common/include/common.h \
             operations/include/Operation.h \
             operations/include/CNOT.h \
             operations/include/U3.h \
             operations/include/Operation_block.h \
             decomposition/include/Decomposition_Base.h \
             decomposition/include/N_Qubit_Decomposition.h \
             decomposition/include/Sub_Matrix_Decomposition.h \
             random_unitary/include/Random_Unitary.h

lib_LTLIBRARIES = libqgd.la
pkginclude_HEADERS = python_interface.h
libqgd_la_SOURCES = python_interface.cpp
AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)
libqgd_la_LIBADD = $(GSL_LIBADD) $(SUBLIBS) #$(GSL_LIB_DIR)/libgsl.la $(GSL_LIB_DIR)/libgslcblas.la

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 
