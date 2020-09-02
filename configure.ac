AC_INIT([qgd], [1.1], [rakytap@caesar.elte.hu])
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
  CXXFLAGS=" -qopenmp $CXXFLAGS"
  CFLAGS=" -qopenmp $CFLAGS"
else
  CXXFLAGS=" -fopenmp $CXXFLAGS"
  CFLAGS=" -fopenmp $CFLAGS"
fi


 

#checking MKL library
MKL_LIBS=
LDFLAGS="$LDFLAGS -L$MKLROOT/lib/intel64/"
AC_CHECK_LIB([mkl_intel_lp64], [zgemm3m], [
MKL_LIBS="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread"
HAVE_LIBMKL_INTEL_LP64=1
], 
[], [-lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread])



#checking MKL headres
CFLAGS_SAVE=$CFLAGS
CFLAGS="$CFLAGS  -I$MKLROOT/include"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$MKLROOT/include"
AC_CHECK_HEADER([mkl_service.h])
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE

#checking GNU SCIENTIFIC LIBRARY
LDFLAGS="${LDFLAGS} -L$GSL_LIB_DIR"
GSL_LIBS=
AC_CHECK_LIB([gsl], [gsl_multimin_fdfminimizer_vector_bfgs2], [
if test $HAVE_LIBMKL_INTEL_LP64; then
GSL_LIBS="-lgsl"
else
GSL_LIBS="-lgsl -lgslcblas "
fi 
], 
AC_MSG_ERROR([gsl library not found]), [-lgslcblas])


LDFLAGS="$LDFLAGS $GSL_LIBS $MKL_LIBS"

#checking GNU SCIENTIFIC HEADERS
CFLAGS_SAVE=$CFLAGS
CFLAGS="$CFLAGS -I$GSL_INC_DIR"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$GSL_INC_DIR"
AC_CHECK_HEADER([gsl/gsl_multimin.h], [], AC_MSG_ERROR([gsl_multimin.h header not found]))
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE



# setting linking flags for libtool
if test $HAVE_LIBMKL_INTEL_LP64 && test $CCX=icpc; then
postdeps_CXX="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread -liomp5"
else
postdeps_CXX="$postdeps_CXX -lgomp"
# checking aligned_alloc if mkl is not present
AC_CHECK_FUNC([aligned_alloc])
fi

CXXFLAGS="$CXXFLAGS -I$GSL_INC_DIR"
CFLAGS="$CFLAGS -I$GSL_INC_DIR -I$MKLROOT"
echo $GSL_LIBS
echo $postdeps_CXX

AC_OUTPUT
