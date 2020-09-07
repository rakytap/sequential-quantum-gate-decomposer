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

#checking floating point precision configuration setup
AC_ARG_WITH(mkl, [  --with-mkl Links against Intel MKL cblas library.],
 [MKL_OPT=yes], [MKL_OPT=no] )

#checking floating point precision configuration setup
AC_ARG_ENABLE(ffast-math, [  --enable-ffast-math Enables g++ compiler optimization on floating point precision. Supported only for standalone applications, the python interface does not support this speedup.],
 [FP_OPT=yes], [FP_OPT=n] )

# TODO GCC OR ICC?
if test $CXX = "icpc"; then
  CXXFLAGS=" -qopenmp -O2"
  CFLAGS=" -qopenmp -O2"
else
if test $FP_OPT = yes; then
  CXXFLAGS=" -fopenmp -O3 -ffast-math"
  CFLAGS=" -fopenmp -O3 -ffast-math"
else
  CFLAGS=" -fopenmp -O3"
  CFLAGS=" -fopenmp -O3"
fi
fi


CBLAS_LIBS= 

#checking MKL library
if test $MKL_OPT = yes ; then
LDFLAGS="$LDFLAGS -L$MKLROOT/lib/intel64/"
AC_CHECK_LIB([mkl_intel_lp64], [zgemm3m], [
CBLAS_LIBS="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread -liomp5"
HAVE_ZGEMM3M=1
], 
[AC_MSG_ERROR([mkl libraries not found])], [-lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread -liomp5])
fi


#checking MKL headers
if test $MKL_OPT = yes; then
CFLAGS_SAVE=$CFLAGS
CFLAGS="$CFLAGS  -I$MKLROOT/include"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$MKLROOT/include"
AC_CHECK_HEADER([mkl_service.h])
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE
fi

#checking GNU SCIENTIFIC LIBRARY
LDFLAGS="${LDFLAGS} -L$GSL_LIB_DIR"
GSL_LIBS=
AC_CHECK_LIB([gsl], [gsl_multimin_fdfminimizer_vector_bfgs2], [
if test $HAVE_ZGEMM3M; then
GSL_LIBS="-lgsl"
if test $MKL_OPT = yes; then
CXXFLAGS="$CXXFLAGS -DMKL" 
fi
else
GSL_LIBS="-lgsl -lgslcblas "
fi 
], 
AC_MSG_ERROR([gsl library not found]), [-lgslcblas])


LDFLAGS="$LDFLAGS $GSL_LIBS $CBLAS_LIBS"

#checking GNU SCIENTIFIC HEADERS
CFLAGS_SAVE=$CFLAGS
CFLAGS="$CFLAGS -I$GSL_INC_DIR"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$GSL_INC_DIR"
AC_CHECK_HEADER([gsl/gsl_multimin.h], [], AC_MSG_ERROR([gsl_multimin.h header not found]))
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE



# setting linking flags for libtool
if test $MKL_OPT = yes; then
postdeps_CXX="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread"
CXXFLAGS="$CXXFLAGS -I$MKLROOT/include"
CFLAGS="$CFLAGS -I$MKLROOT/include"
fi

if test $CXX = g++; then
postdeps_CXX="$postdeps_CXX -lstdc++"
fi

if test $CXX = icpc; then
postdeps_CXX="$postdeps_CXX -liomp5"
else
postdeps_CXX="$postdeps_CXX -lgomp"
fi


CXXFLAGS="$CXXFLAGS -I$GSL_INC_DIR"
CFLAGS="$CFLAGS -I$GSL_INC_DIR"
echo $GSL_LIBS
echo $postdeps_CXX
echo $CXXFLAGS

AC_OUTPUT
