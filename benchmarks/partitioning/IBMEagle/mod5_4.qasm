OPENQASM 2.0;
include "qelib1.inc";

qreg in[5];
x in[4];
h in[4];
h in[4];
ccx in[0],in[3],in[4];
h in[4];
h in[4];
ccx in[2],in[3],in[4];
h in[4];
h in[4];
cx in[3],in[4];
h in[4];
h in[4];
ccx in[1],in[2],in[4];
h in[4];
h in[4];
cx in[2],in[4];
h in[4];
h in[4];
ccx in[0],in[1],in[4];
h in[4];
h in[4];
cx in[1],in[4];
cx in[0],in[4];

