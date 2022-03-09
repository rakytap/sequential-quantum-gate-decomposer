OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(0,2.1277141e-08,-5.4976607) q[3];
u(pi/2,-1.7266761,-11*pi/8) q[2];
cz q[3],q[2];
u(7*pi/8,1.1961793,3.2974724) q[2];
u(pi,-3*pi/2,1.7550691) q[1];
cz q[2],q[1];
u(pi/2,4.3796866,2.7487671) q[3];
u(pi/2,-4.0802721,-pi) q[0];
cz q[3],q[0];
u(pi/2,-2.3484487,0.3746171) q[2];
u(pi/2,-pi/2,7*pi/8) q[0];
cx q[2],q[0];
u(-5*pi/8,1.3903428e-07,-7.8780778e-08) q[0];
cx q[2],q[0];
u(pi/2,-1.0356464,pi/2) q[4];
u(1.4315455,-1.5487775,-4.6907339) q[0];
cz q[4],q[0];
u(0,6.7788833e-08,1.2168929) q[1];
u(pi/2,0.16630449,-0.1768102) q[0];
cz q[1],q[0];
u(pi/2,0.10063225,1.9034987) q[3];
u(pi/2,-0.0051443801,0.14547701) q[1];
cz q[3],q[1];
u(pi,1.8285492,0.96229289) q[2];
u(0.98414648,1.0485999e-07,4.7175333) q[1];
cx q[2],q[1];
u(9*pi/8,1.2057418e-07,1.0995875e-07) q[1];
cx q[2],q[1];
u(pi,0.26336552,-3.7847944) q[4];
u(pi/2,-1.5273837,2.9752882) q[0];
cz q[4],q[0];
u(-8.5811967e-13,-2.2171893,0.15478182) q[3];
u(1.4406726,0.16019444,-0.439646) q[0];
cz q[3],q[0];
u(pi,5.9446869,0.24762261) q[3];
u(pi/2,-3.97392,-5.9635987) q[2];
cz q[3],q[2];
u(5*pi/8,4.2479706,-0.73846891) q[2];
u(pi/2,-3.3932112,3.0356301) q[0];
cz q[2],q[0];
u(pi/2,1.9833778,0.46441833) q[2];
u(0.19395077,1.6397626e-08,-2*pi) q[1];
cz q[2],q[1];
u(pi/2,0.2289992,-1.5906787) q[2];
u(pi/2,0.0039068138,0.64431905) q[0];
cz q[2],q[0];
u(3.0745673,-0.63599938,-0.37140876) q[4];
u(pi/2,pi/2,-0.19755691) q[3];
cz q[4],q[3];
u(pi/2,0.097141702,-0.93478836) q[4];
u(pi/2,-pi,1.0843271e-09) q[3];
cz q[4],q[3];
u(-pi,2.2521222,2.689359) q[4];
u(4.1956899e-08,-1.8226548,1.5936557) q[2];
cz q[4],q[2];
u(pi/2,4.0287606,3*pi/2) q[2];
u(pi/2,-3*pi/2,3.1376858) q[0];
cz q[2],q[0];
u(pi/2,-2.1570718,3*pi/2) q[1];
u(0,-3.0694252e-07,pi) q[0];
cz q[1],q[0];
u(pi/2,-1.0024729,pi) q[3];
u(pi/2,-4.7123878,-4.0287606) q[2];
cz q[3],q[2];
u(0,1.2758315e-07,-0.16151614) q[3];
u(pi/2,-pi,pi) q[0];
cz q[3],q[0];
u(pi/2,7.121493e-09,pi/2) q[0];
u(5.5728516e-09,-1.3415506e-07,0.19357649) q[1];
u(0,1.0717725e-07,-pi/2) q[2];
u(pi,1.4951005e-07,1.2310145) q[3];
u(pi/2,pi,0.53437845) q[4];
