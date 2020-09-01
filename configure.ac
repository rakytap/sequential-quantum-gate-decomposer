AC_INIT([qgd], [1.0], [rakytap@caesar.elte.hu])
AM_INIT_AUTOMAKE
LT_INIT
AC_PROG_CXX
AC_CONFIG_FILES([Makefile 
                 qgd/Makefile
                 common/Makefile
                 operations/Makefile
                 decomposition/Makefile
                 random_unitary/Makefile
                 test_standalone/Makefile])
# TODO GCC OR ICC?
if test $CXX = "icpc"; then
  CXXFLAGS="$CXXFLAGS  -qopenmp -I$GSL_INCLUDE_DIR"
  CFLAGS="$CFLAGS -qopenmp -I$GSL_INCLUDE_DIR"
  LDFLAGS="$LDFLAGS -liomp5"
else
  CXXFLAGS=" -fopenmp $CXXFLAGS -I$GSL_INCLUDE_DIR"
  CFLAGS=" -fopenmp $CFLAGS -I$GSL_INCLUDE_DIR"
  LDFLAGS="$LDFLAGS -lgomp"
fi


 

#checking MKL library
MKL_LIBS=
AC_CHECK_LIB([mkl_intel_lp64], [zgemm3m], [MKL_LIBS="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread"], 
[], [-lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread])

#checking MKL headres
AC_CHECK_HEADER([mkl_service.h])


#checking GNU SCIENTIFIC LIBRARY
LDFLAGS="${LDFLAGS_save} -L$GSL_LIB_DIR"
GSL_LIBS=
AC_CHECK_LIB([gsl], [gsl_multimin_fdfminimizer_vector_bfgs2], [
if $ac_cv_libs_mkl_intel_lp64; then
GSL_LIBS="-lgsl"
else
GSL_LIBS="-lgsl -lgslcblas "
fi 
], 
AC_MSG_ERROR([gsl library not found]), [-lgslcblas])


LDFLAGS="$LDFLAGS $GSL_LIBS $MKL_LIBS"

#checking GNU SCIENTIFIC HEADERS
AC_CHECK_HEADER([gsl/gsl_multimin.h], [], AC_MSG_ERROR([gsl_multimin.h header not found]))

if $ac_cv_libs_mkl_intel_lp64; then
postdeps_CXX="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread"
else
# checking aligned_alloc if mkl is not present
AC_CHECK_FUNC([aligned_alloc])
fi


AC_OUTPUT
