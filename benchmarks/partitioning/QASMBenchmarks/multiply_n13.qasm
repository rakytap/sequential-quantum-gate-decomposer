OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[4];
// This initializes 13 quantum registers and 4 classical registers.

x q[0];
x q[1]; // 1st 2 qubits 11 (3)
x q[2];
x q[4]; // next 3 qubits 101 (5)
// All qubits start at a ground state of 0; this changes 4 of the first 5 qubits to 1 so that I could use a binary 11 (digital 3) and a binary 101 (digital 5).

ccx q[2], q[0], q[5]; // LSQ
ccx q[2], q[1], q[6];
ccx q[3], q[0], q[7];
ccx q[3], q[1], q[8];
ccx q[4], q[0], q[9];
ccx q[4], q[1], q[10]; // MSQ
// Multiplication is all AND gates: 1 x 1 = 1; all else is 0.

cx q[6], q[11];
cx q[7], q[11]; // 2nd digit
cx q[8], q[12];
cx q[9], q[12]; // 3rd digit
// With 3 x 5, all addition can be done with simple XOR gates.


