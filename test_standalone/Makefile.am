#test_standalone/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

bin_PROGRAMS = decomposition_test	

#pkginclude_HEADERS = include/Random_Unitary.h
		
decomposition_test_SOURCES = decomposition_test.cpp

AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)

#decomposition_test_LDFLAGS = $(top_srcdir)/random_unitary/.libs/librandomunitary.a  $(top_srcdir)/decomposition/.libs/libdecomposition.a $(top_srcdir)/operations/.libs/liboperations.a $(top_srcdir)/common/.libs/libcommon.a $(GSL_LIB_DIR)/libgsl.a $(GSL_LIB_DIR)/libgslcblas.a  -liomp5  
decomposition_test_LDFLAGS = $(top_srcdir)/random_unitary/.libs/librandomunitary.a $(top_srcdir)/decomposition/.libs/libdecomposition.a $(top_srcdir)/operations/.libs/liboperations.a $(top_srcdir)/common/.libs/libcommon.a 
