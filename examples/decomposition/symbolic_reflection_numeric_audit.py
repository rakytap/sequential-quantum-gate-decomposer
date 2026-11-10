#!/usr/bin/env python3
"""Reproduce seven-CX numerical reflection witnesses; not family proofs."""
import argparse,hashlib,importlib.metadata
from pathlib import Path
import sys,json,gzip,time,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'squander/decomposition'))
from symbolic_gates import make_gate
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate,U3Gate
layout=((0,1),(1,2),(0,1),(1,2),(0,2),(1,2),(1,2))
initial=Circuit(3)
for q in range(3):initial.append_gate(U3Gate(),q)
for edge in layout:
    initial.append_gate(CNOTGate(),edge)
    for q in edge:initial.append_gate(U3Gate(),q)
v=np.array([np.cos(np.pi/8),0,0,np.sin(np.pi/8)]);T=np.eye(8,dtype=complex);T[4:,4:]=np.eye(4)-2*np.outer(v,v)
initial.instantiate(T,multistarts=2,seed=776,method='minimization')
ops=[dict(gate=str(o.gate),wires=list(o.location),parameters=list(o.params)) for o in initial]
p0=np.array([v for o in ops for v in o['parameters']])

def embed(m,wires):
 U=np.zeros((8,8),complex)
 for i in range(8):
  j=sum(((i>>q)&1)<<k for k,q in enumerate(wires));base=i
  for q in wires:base&=~(1<<q)
  for x in range(m.shape[0]):
   y=base
   for k,q in enumerate(wires):y|=((x>>k)&1)<<q
   U[y,i]=m[x,j]
 return U
# BQSKit gate matrices use the first location as the most significant.
CX=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
def u3(p):
 t,f,l=p;c,s=np.cos(t/2),np.sin(t/2)
 return np.array([[c,-np.exp(1j*l)*s],[np.exp(1j*f)*s,np.exp(1j*(f+l))*c]])
def word(p):
 out=[];k=0
 for o in ops:
  out.append(dict(gate='U3' if o['parameters'] else 'CX',wires=o['wires'],parameters=list(p[k:k+3]) if o['parameters'] else []))
  if o['parameters']:k+=3
 return out
def matrix(w):
 U=np.eye(8,dtype=complex)
 for o in w:
  m=u3(o['parameters']) if o['parameters'] else CX
  # big endian wire numbering, use reversed location for multiwire input map.
  wires=[2-q for q in o['wires']]
  if len(wires)==2:wires=wires[::-1]
  U=embed(m,wires)@U
 return U
def target(t):
 v=np.array([np.cos(t),0,0,np.sin(t)]);R=np.eye(4)-2*np.outer(v,v);T=np.eye(8,dtype=complex);T[4:,4:]=R
 return T
rows=[]
for angle in [np.pi/8,.3,0,.1,.2,.4,.5,.6,.7,np.pi/4]:
 T=target(angle)
 def residual(p):
  U=matrix(word(p));phase=np.vdot(U.ravel(),T.ravel());phase/=abs(phase);D=phase*U-T
  return np.r_[D.real.ravel(),D.imag.ravel()]
 result=least_squares(residual,p0,max_nfev=500,ftol=1e-13,xtol=1e-13,gtol=1e-13)
 p0=result.x;w=word(p0);U=matrix(w);phase=np.vdot(U.ravel(),T.ravel());phase/=abs(phase);gamma=np.angle(phase)
 t,p,l=w[0]['parameters'];w=[dict(gate='U3',wires=w[0]['wires'],parameters=[np.pi-t,0,l-np.pi]),dict(gate='U3',wires=w[0]['wires'],parameters=[np.pi,gamma+p,np.pi+gamma])]+w[1:]
 err=float(np.max(np.abs(matrix(w)-T)));print(angle,err,flush=True)
 rows.append(dict(angle=repr(angle),angle_source='pi/8' if angle==np.pi/8 else 'pi/4' if angle==np.pi/4 else repr(angle),entanglers=7,literal_maximum_entry_error=err,operations=w))
with gzip.open(args.output,'wt') as f:json.dump(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','bqskit')},settings=dict(seed=776,multistarts=2,layout=layout,max_nfev=500),schema='reflection-seven-CNOT-numerical-witnesses-v1',rows=rows,wire_order='BQSKit big endian c,a,b',contract='Literal numeric replay, no GP, 7 CX and U3 locals. No exact family identity or minimum proof.'),f)
