/*
This code is credited to the OPTIMUS project of Ioannis G. Tsoulos
*/

#ifndef LBFGS_H
#define LBFGS_H
# include "tolmin.h"

typedef long int ftnlen;
typedef long int logical;

#include <math.h>
#include <time.h>

#ifndef FTOL
#define FTOL .001
#endif
#ifndef GTOL
#define GTOL .9
#endif
#ifndef XTOL
#define XTOL .1
#endif
#ifndef STEPMIN
#define STEPMIN 0.
#endif
#if defined(_WIN32) || defined(__hpux) || !defined(__GFORTRAN__)
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
/* if we're not WIN32 or HPUX, then if we are linking
 * with gfortran (instead of gcc or g++), we need to mangle
 * names appropriately */
#endif


/* If we want machine precision in a nice fashion, do this: */
#include <float.h>
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2e-16
#endif




#define TRUE_ (1)
#define FALSE_ (0)
#define START 1
#define NEW_X 2
#define ABNORMAL 3 /* message: ABNORMAL_TERMINATION_IN_LNSRCH. */
#define RESTART 4 /* message: RESTART_FROM_LNSRCH. */

#define FG      10
#define FG_END  15
#define IS_FG(x) ( ((x)>=FG) ?  ( ((x)<=FG_END) ? 1 : 0 ) : 0 )
#define FG_LN   11
#define FG_LNSRCH FG_LN
#define FG_ST   12
#define FG_START FG_ST

#define CONVERGENCE 20
#define CONVERGENCE_END  25
#define IS_CONVERGED(x) ( ((x)>=CONVERGENCE) ?  ( ((x)<=CONVERGENCE_END) ? 1 : 0 ) : 0 )
#define CONV_GRAD   21 /* message: CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL. */
#define CONV_F      22 /* message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH. */

#define STOP  30
#define STOP_END 40
#define IS_STOP(x) ( ((x)>=STOP) ?  ( ((x)<=STOP_END) ? 1 : 0 ) : 0 )
#define STOP_CPU  31 /* message: STOP: CPU EXCEEDING THE TIME LIMIT. */
#define STOP_ITER 32 /* message: STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIM.  */
#define STOP_GRAD 33 /* message: STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL. */

#define WARNING 100
#define WARNING_END 110
#define IS_WARNING(x) ( ((x)>=WARNING) ?  ( ((x)<=WARNING_END) ? 1 : 0 ) : 0 )
#define WARNING_ROUND 101  /* WARNING: ROUNDING ERRORS PREVENT PROGRESS */
#define WARNING_XTOL  102  /* WARNING: XTOL TEST SATISIED */
#define WARNING_STPMAX 103 /* WARNING: STP = STPMAX */
#define WARNING_STPMIN 104 /* WARNING: STP = STPMIN */

#define ERROR 200
#define ERROR_END 240
#define IS_ERROR(x) ( ((x)>=ERROR) ?  ( ((x)<=ERROR_END) ? 1 : 0 ) : 0 )
/* More specific conditions below */
#define ERROR_SMALLSTP 201 /* message: ERROR: STP .LT. STPMIN  */
#define ERROR_LARGESTP 202 /* message: ERROR: STP .GT. STPMAX  */
#define ERROR_INITIAL  203 /* message: ERROR: INITIAL G .GE. ZERO */
#define ERROR_FTOL     204 /* message: ERROR: FTOL .LT. ZERO   */
#define ERROR_GTOL     205 /* message: ERROR: GTOL .LT. ZERO   */
#define ERROR_XTOL     206 /* message: ERROR: XTOL .LT. ZERO   */
#define ERROR_STP0     207 /* message: ERROR: STPMIN .LT. ZERO */
#define ERROR_STP1     208 /* message: ERROR: STPMAX .LT. STPMIN */
#define ERROR_N0       209 /* ERROR: N .LE. 0 */
#define ERROR_M0       210 /* ERROR: M .LE. 0 */
#define ERROR_FACTR    211 /* ERROR: FACTR .LT. 0 */
#define ERROR_NBD      212 /* ERROR: INVALID NBD */
#define ERROR_FEAS     213 /* ERROR: NO FEASIBLE SOLUTION */


/* and "word" was a char that was one fo these: */
#define WORD_DEFAULT 0 /* aka "---".  */
#define WORD_CON 1 /*  the subspace minimization converged. */
#define WORD_BND 2 /* the subspace minimization stopped at a bound. */
#define WORD_TNT 3 /*  the truncated Newton step has been used. */

/** @brief This class is used to implement the L-BFGS local optimizer **/
class Lbfgs
{
private:
    Problem *myProblem;
    int prn1lb(integer *n, integer *m, double *l,
        double *u, double *x, integer *iprint, FILE* itfile,
        double *epsmch);

    int prn2lb(integer *n, double *x, double *f,
       double *g, integer *iprint, FILE* itfile, integer *iter,
       integer *nfgv, integer *nact, double *sbgnrm, integer *nseg, integer
       *word, integer *iword, integer *iback, double *stp, double *
       xstep, ftnlen word_len);


    int daxpy(integer *n, double *da, double *dx,
        integer *incx, double *dy, integer *incy);

    int prn3lb(integer *n, double *x, double *f, integer *
        task, integer *iprint, integer *info, FILE* itfile, integer *iter,
        integer *nfgv, integer *nintol, integer *nskip, integer *nact,
        double *sbgnrm, double *time, integer *nseg, integer *word,
        integer *iback, double *stp, double *xstep, integer *k,
        double *cachyt, double *sbtime, double *lnscht, ftnlen
        task_len, ftnlen word_len);

    double ddot(integer *n, double *dx, integer *incx, double *dy,
            integer *incy);
    int dscal(integer *n, double *da, double *dx,
            integer *incx);
    int hpsolb(integer *n, double *t, integer *iorder,
            integer *iheap);
    int bmv(integer *m, double *sy, double *wt, integer
        *col, double *v, double *p, integer *info);
    int dcopy(integer *, double *, integer *,
            double *, integer *);
    int setulb(integer *n, integer *m, double *x,
        double *l, double *u, integer *nbd, double *f, double
        *g, double *factr, double *pgtol, double *wa, integer *
        iwa, integer *task, integer *iprint, integer *csave, logical *lsave,
        integer *isave, double *dsave);
    int mainlb(integer *n, integer *m, double *x,
            double *l, double *u, integer *nbd, double *f, double
            *g, double *factr, double *pgtol, double *ws, double *
            wy, double *sy, double *ss, double *wt, double *wn,
            double *snd, double *z__, double *r__, double *d__,
            double *t, double *xp, double *wa, integer *index,
            integer *iwhere, integer *indx2, integer *task, integer *iprint,
            integer *csave, logical *lsave, integer *isave, double *dsave);
    int freev(integer *, integer *, integer *,
           integer *, integer *, integer *, integer *, logical *, logical *,
           logical *, integer *, integer *);
    int timer(double *);
    int formk(integer *,
            integer *, integer *, integer *, integer *, integer *, integer *,
            logical *, double *, double *, integer *, double *,
            double *, double *, double *, integer *, integer *,
            integer *);
    int formt(integer *, double *, double *,
            double *, integer *, double *, integer *);
    int subsm(integer *, integer *, integer *, integer *, double *, double *,
           integer *, double *, double *, double *, double *,
           double *, double *, double *, double *, integer *
           , integer *, integer *, double *, double *, integer *,
           integer *);

    int errclb(integer *n, integer *m, double *factr,
            double *l, double *u, integer *nbd, integer *task, integer *info,
            integer *k, ftnlen task_len);
    int active(integer *, double *, double *,
           integer *, double *, integer *, integer *, logical *,
           logical *, logical *);
    int cauchy(integer *, double *,
            double *, double *, integer *, double *, integer *,
            integer *, double *, double *, double *, integer *,
            double *, double *, double *, double *,
            double *, integer *, integer *, double *, double *,
            double *, double *, integer *, integer *, double *,
            integer *, double *);
    int cmprlb(integer *, integer *, double *,
            double *, double *, double *, double *,
            double *, double *, double *, double *, integer *,
            double *, integer *, integer *, integer *, logical *,
            integer *);
    int matupd(integer *, integer *, double *,
            double *, double *, double *, double *,
            double *, integer *, integer *, integer *, integer *,
            double *, double *, double *, double *,
            double *);
    int lnsrlb(integer *n, double *l, double *u,
            integer *nbd, double *x, double *f, double *fold,
            double *gd, double *gdold, double *g, double *d__,
            double *r__, double *t, double *z__, double *stp,
            double *dnorm, double *dtd, double *xstep, double *
            stpmx, integer *iter, integer *ifun, integer *iback, integer *nfgv,
            integer *info, integer *task, logical *boxed, logical *cnstnd, integer *
            csave, integer *isave, double *dsave); /* ftnlen task_len, ftnlen
            csave_len); */
    int projgr(integer *, double *, double *,
           integer *, double *, double *, double *);
    int dcsrch(double *f, double *g, double *stp,
            double *ftol, double *gtol, double *xtol, double *
            stpmin, double *stpmax, integer *task, integer *isave, double *
            dsave);
    int dcstep(double *, double *,
            double *, double *, double *, double *,
            double *, double *, double *, logical *, double *,
            double *);
    int dpofa(double *, integer *, integer *,
           integer *);
    int  dtrsl(double *, integer *, integer *,
           double *, integer *, integer *);
public:
    Lbfgs(Problem *p);
    double Solve(Data &x);
};

#endif // LBFGS_H
