OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(pi/2,4.0162442,-2.1571597) q[2];
u(pi,-3*pi/2,3.1238913) q[0];
cz q[2],q[0];
u(0,1.4164572e-08,4.6634844) q[2];
u(pi,-3*pi/2,-5.9896897) q[1];
cz q[2],q[1];
u(pi/2,1.4275491,-7.7835533e-09) q[3];
u(pi/2,-pi/2,5.4977824) q[0];
cx q[3],q[0];
u(pi/4,4.8401366e-07,-1.9621264e-09) q[0];
cx q[3],q[0];
u(9.9364743e-09,-1.4342594,-0.32103034) q[3];
u(pi/2,-pi/2,3*pi/4) q[1];
cx q[3],q[1];
u(-3*pi/4,-1.1707671e-08,-2.2814852e-08) q[1];
cx q[3],q[1];
u(-4.7030082e-09,-1.1556138e-08,4.2926855) q[3];
u(pi/4,6.476224e-08,5.4574382) q[2];
cx q[3],q[2];
u(-pi/4,2.7764323e-08,2.9941597e-08) q[2];
cx q[3],q[2];
u(pi/2,-3.3323086,-pi) q[2];
u(pi/2,-3.1592889,-pi/2) q[0];
cz q[2],q[0];
u(pi/2,0.097229981,pi) q[4];
u(pi/2,-pi/2,1.7615124) q[2];
cz q[4],q[2];
u(pi/2,3.2842995,pi/2) q[2];
u(pi/2,-0.29349568,-3*pi/2) q[1];
cz q[2],q[1];
u(-pi,1.7072882,3.0046941) q[3];
u(2.9425579,-4.1960146e-08,1.4280894) q[2];
cx q[3],q[2];
u(-pi/4,9.015819e-09,-1.9621003e-09) q[2];
cx q[3],q[2];
u(pi,2.0904367,-2.0447632) q[4];
u(pi,-pi,pi) q[0];
cz q[4],q[0];
u(pi,pi,pi/2) q[0];
u(pi,-2.4794309e-08,pi/2) q[1];
u(pi/2,-3.0127232e-08,3*pi/2) q[2];
u(pi/2,-2.5744444e-09,-2.5916309) q[3];
u(pi/2,pi,-0.89637727) q[4];