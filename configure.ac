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
AC_ARG_ENABLE(fp-optimization, [  --enable-fp-optimization Enables compiler optimization on floating point precision. Supported only for standalone applications, the python interface does not support this speedup.],
 [FP_OPT=yes], [] )

# TODO GCC OR ICC?
if test $CXX = "icpc"; then
if test $FP_OPT; then
  CXXFLAGS=" -qopenmp -O3 -falign-loops=64"
  CFLAGS=" -qopenmp -O3 -falign-loops=64"
else
  CXXFLAGS="-mieee-fp -qopenmp -O3 -falign-loops=64"
  CFLAGS="-mieee-fp -qopenmp -O3 -falign-loops=64"  
fi
else
if test $FP_OPT; then
  CXXFLAGS=" -fopenmp -O3 -ffast-math"
  CFLAGS=" -fopenmp -O3 -ffast-math"
else
  CFLAGS=" -fopenmp -O3"
  CFLAGS=" -fopenmp -O3"
fi
fi


 

#checking MKL library
MKL_LIBS=
LDFLAGS="$LDFLAGS -L$MKLROOT/lib/intel64/"
AC_CHECK_LIB([mkl_intel_lp64], [zgemm3m], [
MKL_LIBS="-lmkl_intel_lp64 -lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread -liomp5"
HAVE_LIBMKL_INTEL_LP64=1
], 
[], [-lmkl_core -lpthread -lm -ldl -lmkl_def -lmkl_intel_thread -liomp5])



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
if test CXX=g++; then
postdeps_CXX="$postdeps_CXX -lstdc++"
fi
else
postdeps_CXX="$postdeps_CXX -lgomp"
# checking aligned_alloc if mkl is not present
AC_CHECK_FUNC([aligned_alloc])
fi

CXXFLAGS="$CXXFLAGS -I$GSL_INC_DIR"
CFLAGS="$CFLAGS -I$GSL_INC_DIR -I$MKLROOT"
echo $GSL_LIBS
echo $postdeps_CXX
echo $CXXFLAGS

AC_OUTPUT
