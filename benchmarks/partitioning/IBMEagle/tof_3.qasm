OPENQASM 2.0;
include "qelib1.inc";

qreg in[5];
h in[4];
h in[4];
ccx in[0],in[1],in[4];
h in[4];
h in[4];
h in[3];
h in[3];
ccx in[2],in[4],in[3];
h in[3];
h in[3];
h in[4];
h in[4];
ccx in[0],in[1],in[4];
h in[4];
h in[4];

