AC_INIT([qgd], [1.2], [rakytap@caesar.elte.hu])
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
AC_ARG_WITH(mkl, [  --with-mkl Links against Intel MKL library.],
 [MKL_OPT=yes], [MKL_OPT=no] )

#checking floating point precision configuration setup
AC_ARG_WITH(tbb, [  --with-tbb Link and compile against TBB library instead of OpenMP.],
 [TBB_OPT=yes], [TBB_OPT=no] )

#checking floating point precision configuration setup
AC_ARG_ENABLE(ffast-math, [  --enable-ffast-math Enables g++ compiler optimization on floating point precision. Supported only for standalone applications, the python interface does not support this speedup.],
 [FP_OPT=yes], [FP_OPT=n] )

# Ensure that tests are run with the C++ compiler
AC_LANG([C++])

# TODO GCC OR ICC?
# INTEL compiler
if test $CXX = "icpc"; then
if test $TBB_OPT = yes ; then
  CXXFLAGS="-O2 -tbb"
else
  CXXFLAGS="-O2 -qopenmp"
fi

else

# GNU compiler
if test $FP_OPT = yes; then
  CXXFLAGS="-O3 -ffast-math -ftree-vectorize"
else
  CXXFLAGS="-O2 -ftree-vectorize"
fi

if test $TBB_OPT = yes ; then
  CXXFLAGS="$CXXFLAGS"
else
  CXXFLAGS="$CXXFLAGS -fopenmp"
fi

fi


#checking TBB library
if test $TBB_OPT = yes ; then
CXXFLAGS="$CXXFLAGS -DTBB"
LDFLAGS="${LDFLAGS}"
if test x$TBB_LIB_DIR != x ; then
LDFLAGS="${LDFLAGS} -L$TBB_LIB_DIR"
fi

TBB_LIBS=""
AC_CHECK_LIB([tbb], [pthread_attr_init], [
if test $CXX = g++; then
TBB_LIBS="-ltbb"
fi
], 
AC_MSG_ERROR([TBB library not found]), [])
fi

#checking TBB HEADERS
if test $TBB_OPT = yes ; then
CPPFLAGS_SAVE=$CPPFLAGS
if test x$TBBROOT != x ; then
CPPFLAGS="$CPPFLAGS  -I$TBBROOT/include"
fi
if test x$TBB_INC_DIR != x ; then
CPPFLAGS="$CPPFLAGS  -I$TBB_INC_DIR"
fi

AC_CHECK_HEADER([tbb/tbb.h], [], AC_MSG_ERROR([tbb.h header not found]))
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE
fi


#checking MKL library
if test $MKL_OPT = yes ; then
CXXFLAGS="$CXXFLAGS -DMKL -mkl"
AC_CHECK_LIB([mkl_intel_lp64], [zgemm3m], [
HAVE_ZGEMM3M=1
], 
[AC_MSG_ERROR([mkl libraries not found])], [])
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
if test x$GSL_LIB_DIR != x ; then
LDFLAGS="${LDFLAGS} -L$GSL_LIB_DIR"
fi
GSL_LIBS=
AC_CHECK_LIB([gsl], [gsl_multimin_fdfminimizer_vector_bfgs2], [
if test $HAVE_ZGEMM3M; then
GSL_LIBS="-lgsl"
else
GSL_LIBS="-lgsl -lgslcblas "
fi 
], 
AC_MSG_ERROR([gsl library not found]), [-lgslcblas])



#checking GNU SCIENTIFIC HEADERS
CFLAGS_SAVE=$CFLAGS
CFLAGS="$CFLAGS -I$GSL_INC_DIR"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$GSL_INC_DIR"
AC_CHECK_HEADER([gsl/gsl_multimin.h], [], AC_MSG_ERROR([gsl_multimin.h header not found]))
CFLAGS=$CFLAGS_SAVE
CPPFLAGS=$CPPFLAGS_SAVE




# setting linking flags for libtool
LDFLAGS="$LDFLAGS $GSL_LIBS $TBB_LIBS"


# adding MKL include paths
if test $MKL_OPT = yes; then
if test x$MKLROOT != x ; then
CXXFLAGS="$CXXFLAGS -I$MKLROOT/include"
fi
fi

# adding TBB include paths
if test $TBB_OPT = yes; then
if test x$TBBROOT != x ; then
CXXFLAGS="$CXXFLAGS -I$TBBROOT/include"
fi
if test x$TBB_INC_DIR != x ; then
CXXFLAGS="$CXXFLAGS -I$TBB_INC_DIR"
fi
fi

# adding GSL include paths
if test x$GSL_INC_DIR != x ; then
CXXFLAGS="$CXXFLAGS -I$GSL_INC_DIR"
fi

if test $CXX = g++; then
postdeps_CXX="$postdeps_CXX -lstdc++"
fi

if test $CXX = icpc; then
postdeps_CXX=""
else
postdeps_CXX="$postdeps_CXX $TBB_LIBS"
fi


CXXFLAGS="$CXXFLAGS -g -Wall"
echo "LDFLAGS:" $LDFLAGS
echo "postdeps:" $postdeps_CXX
echo "CXX flags:" $CXXFLAGS

AC_OUTPUT
