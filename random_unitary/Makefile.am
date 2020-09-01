#random_unitary/Makefile.am

all:
	@echo
	@echo "----------------------------------------------------------"
	@echo 
	@echo "----------------------------------------------------------"
	@echo 

noinst_LTLIBRARIES	= librandomunitary.la

pkginclude_HEADERS = include/Random_Unitary.h

librandomunitary_la_SOURCES	= Random_Unitary.cpp

AM_CPPFLAGS = $(CXXFLAGS) -I$(top_srcdir) -I$(GSL_INC_DIR)
