OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(-1.10325e-09,1.4984696e-09,-0.46346379) q[1];
u(pi/4,1.1062765,-pi/2) q[0];
cz q[1],q[0];
u(6.6160775,1.7795359,pi/2) q[4];
u(pi/2,-3.9171511,0.46451989) q[0];
cz q[4],q[0];
u(pi,3.2161623,0.54975136) q[2];
u(pi/2,-1.5707824,-0.0098397619) q[0];
cz q[2],q[0];
u(-pi,-0.18499791,2.2643324) q[4];
u(pi,-3*pi/2,-3.5273338) q[1];
cz q[4],q[1];
u(0,2.6243604e-09,1.137129) q[1];
u(3*pi/4,-3.141601,3.1415787) q[0];
cz q[1],q[0];
u(pi/2,-1.1227965,-3.6251154) q[4];
u(3*pi/4,-3.1415809,3.1416009) q[0];
cz q[4],q[0];
u(pi,-1.2508825,1.2570305) q[2];
u(-4.9666495e-09,-1.5708022,4.7123831) q[0];
cz q[2],q[0];
u(-2.1359388e-08,9.5531211e-09,-1.8295674) q[2];
u(pi/2,pi/2,0.36986054) q[1];
cx q[2],q[1];
u(3*pi/4,1.7615505e-09,1.7158428e-09) q[1];
cx q[2],q[1];
u(0.33289219,pi/2,pi) q[0];
u(pi/2,pi,3*pi/2) q[1];
u(9.5165298e-09,9.5531214e-09,-3.2062848) q[2];
u(0,1.7832464e-08,-9.5090344e-09) q[3];
u(pi/2,pi,1.1227965) q[4];
