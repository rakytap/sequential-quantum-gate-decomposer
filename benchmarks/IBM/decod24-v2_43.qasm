OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
u(pi,-4.1079047,-2.0823741) q[3];
u(pi/2,-pi/2,-7.1284863e-09) q[1];
cz q[3],q[1];
u(pi/2,-4.9931098,-2.6868585) q[3];
u(pi/2,-1*pi/2,-3*pi/2) q[0];
cz q[3],q[0];
u(pi,1.6322197,5.3719204) q[2];
u(2.3896946,-1.0070947,3.0695879) q[0];
cx q[2],q[0];
u(-pi/3,3.0969169e-07,1.078205e-07) q[0];
cx q[2],q[0];
u(4.0969093,-2.2817531,-0.50467738) q[3];
u(pi/2,pi/2,1.4345193) q[2];
cx q[3],q[2];
u(-2*pi/3,-7.8963395e-08,-1.0826712e-07) q[2];
cx q[3],q[2];
u(-0.95531665,-1.8872,0.18735799) q[3];
u(2.3202734,-1.9490961,5.3745239) q[0];
cz q[3],q[0];
u(pi/2,1.1862886,5.8141908) q[3];
u(0.83641122,-4.3381793e-08,-pi) q[2];
cz q[3],q[2];
u(pi,-0.08859688,0.77943911) q[3];
u(pi,-pi,pi) q[1];
cz q[3],q[1];
u(pi/2,pi/2,-1.1253224) q[0];
u(pi/2,-4.3082374e-09,3*pi/2) q[1];
u(pi/2,3.3088701e-09,pi/2) q[2];
u(0,-1.0634729e-07,-2.6580642) q[3];