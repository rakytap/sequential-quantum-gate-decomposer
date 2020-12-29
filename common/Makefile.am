#common/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

noinst_LTLIBRARIES	= libcommon.la

pkginclude_HEADERS = include/common.h

libcommon_la_SOURCES	= common.cpp matrix.cpp

AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir)  -I$(GSL_INC_DIR)
