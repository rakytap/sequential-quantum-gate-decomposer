AC_INIT([qgd], [1.3], [rakytap@caesar.elte.hu])
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

#checking whether to use Intel MKL
AC_ARG_WITH(mkl, [  --with-mkl Links against Intel MKL library.],
 [MKL_OPT=yes], [MKL_OPT=no] )

#checking whether to use OpenBLAS
AC_ARG_WITH(openblas, [  --with-openblas Links against OpenBLAS library. If given, option --with-mkl has no effect.],
 [OBLAS_OPT=yes], [OBLAS_OPT=no] )
if test $OBLAS_OPT = yes; then
 MKL_OPT=no
fi

#checking floating point precision configuration setup
AC_ARG_ENABLE(ffast-math, [  --enable-ffast-math Enables g++ compiler optimization on floating point precision. Supported only for standalone applications, the python interface does not support this speedup.],
 [FP_OPT=yes], [FP_OPT=n] )

# Ensure that tests are run with the C++ compiler
AC_LANG([C++])


# INTEL compiler
if test $CXX = "icpc"; then
  CXXFLAGS="$CXXFLAGS -O2 -tbb -qopenmp -Wall -O3 -m64 -msse3 -mssse3 -msse4.1 -mavx2 -DNDEBUG -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fpermissive"

else

# GNU compiler
  CXXFLAGS="$CXXFLAGS -ftree-vectorize -fopenmp -Wall -O3 -m64 -msse3 -mssse3 -msse4.1 -mavx2 -DNDEBUG -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fpermissive"
if test $FP_OPT = yes; then
  CXXFLAGS="$CXXFLAGS -ffast-math"
fi


fi

################################################################################
# checking TBB library
LDFLAGS="${LDFLAGS}  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free"
if test x$TBB_LIB_DIR != x ; then
LDFLAGS="${LDFLAGS} -Wl,-rpath=$TBB_LIB_DIR  -L$TBB_LIB_DIR"
fi

TBB_LIBS=""
AC_CHECK_LIB([tbb], [pthread_attr_init], [
TBB_LIBS="-ltbb"
], 
AC_MSG_ERROR([libtbb library not found]), [])

AC_CHECK_LIB([tbbmalloc], [scalable_aligned_malloc], [
TBB_LIBS="$TBB_LIBS -ltbbmalloc"
], 
AC_MSG_ERROR([libtbbmalloc library not found]), [])

AC_CHECK_LIB([tbbmalloc_proxy], [_init], [
TBB_LIBS="$TBB_LIBS -ltbbmalloc_proxy"
], 
AC_MSG_ERROR([libtbb library not found]), [])

################################################################################
#checking TBB HEADERS
CPPFLAGS_SAVE=$CPPFLAGS
if test x$TBBROOT != x ; then
CPPFLAGS="$CPPFLAGS  -I$TBBROOT/include"
fi
if test x$TBB_INC_DIR != x ; then
CPPFLAGS="$CPPFLAGS  -I$TBB_INC_DIR"
fi

AC_CHECK_HEADER([tbb/tbb.h], [], AC_MSG_ERROR([tbb.h header not found]))
AC_CHECK_HEADER([tbb/scalable_allocator.h], [], AC_MSG_ERROR([scalable_allocator.h header not found]))
CPPFLAGS=$CPPFLAGS_SAVE

################################################################################
#checking MKL library
if test $MKL_OPT = yes ; then
CXXFLAGS="$CXXFLAGS -DCBLAS=1"

if test $CXX = "icpc"; then
  CXXFLAGS="$CXXFLAGS -mkl"
fi

if test x$MKL_LIB_DIR != x ; then
LDFLAGS="${LDFLAGS} -L$MKL_LIB_DIR"
fi

if test $CXX = "g++"; then
  LDFLAGS="${LDFLAGS} -L$MKL_LIB_DIR -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
fi

AC_CHECK_LIB([mkl_intel_lp64], [cblas_zgemm3m], [
HAVE_ZGEMM3M=1
], 
[AC_MSG_ERROR([MKL libraries not found])], [])

fi

################################################################################
#checking MKL headers
if test $MKL_OPT = yes; then
CFLAGS="$CFLAGS  -I$MKLROOT/include"
CPPFLAGS_SAVE=$CPPFLAGS
CPPFLAGS="$CPPFLAGS  -I$MKLROOT/include"
AC_CHECK_HEADER([mkl_service.h])
CPPFLAGS=$CPPFLAGS_SAVE
fi



################################################################################
#checking OpenBLAS library
OBLAS_LIBS=""
if test $OBLAS_OPT = yes ; then
CXXFLAGS="$CXXFLAGS -DCBLAS=2"

if test x$OPENBLAS_LIB_DIR != x ; then
LDFLAGS="${LDFLAGS} -L$OPENBLAS_LIB_DIR"
fi

AC_CHECK_LIB([openblas], [cblas_zgemm3m], [
HAVE_ZGEMM3M=1
OBLAS_LIBS="-lopenblas"
], 
[AC_MSG_ERROR([OpenBLAS libraries not found])], [])
fi


################################################################################
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


################################################################################
#checking GNU SCIENTIFIC HEADERS
CFLAGS="$CFLAGS -I$GSL_INC_DIR"
CPPFLAGS_SAVE=$CPPFLAGS
if test x$GSL_INC_DIR != x ; then
CPPFLAGS="$CPPFLAGS  -I$GSL_INC_DIR"
fi
AC_CHECK_HEADER([gsl/gsl_multimin.h], [], AC_MSG_ERROR([gsl_multimin.h header not found]))
CPPFLAGS=$CPPFLAGS_SAVE




# setting linking flags for libtool
LDFLAGS="$LDFLAGS $GSL_LIBS $TBB_LIBS $OBLAS_LIBS"

# adding TBB include paths
if test x$TBBROOT != x ; then
CXXFLAGS="$CXXFLAGS -I$TBBROOT/include"
fi
if test x$TBB_INC_DIR != x ; then
CXXFLAGS="$CXXFLAGS -I$TBB_INC_DIR"
fi

# adding GSL include paths
if test x$GSL_INC_DIR != x ; then
CXXFLAGS="$CXXFLAGS -I$GSL_INC_DIR"
fi

postdeps_CXX="$postdeps_CXX $GSL_LIBS $TBB_LIBS $OBLAS_LIBS"


CXXFLAGS="$CXXFLAGS -g -Wall"
echo "LDFLAGS:" $LDFLAGS
echo "postdeps:" $postdeps_CXX
echo "CXX flags:" $CXXFLAGS

AC_OUTPUT
