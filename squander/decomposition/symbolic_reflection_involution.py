"""Schmidt-angle normal form for the trace +/-2 involution strata.

These are exact constructions and partial bounds, not a classification of
the optimal middle-stratum cost. Signs change only a local control Z.
"""
import sympy as sp
if __package__:
    from . import symbolic_gates as gates
    from .symbolic_rules import source_rule
else:
    import symbolic_gates as gates
    from symbolic_rules import source_rule


def reflection(theta):
    psi=sp.Matrix([sp.cos(theta),0,0,sp.sin(theta)])
    return sp.ImmutableMatrix(sp.eye(4)-2*psi*psi.H)


def controlled_reflection_source(theta,negative=False):
    """Eight-CX literal word for the complete real-angle family."""
    ccx,factor=source_rule('CCX',())
    if factor!=1:raise ArithmeticError('CCX source must be literal')
    # Controlled R_00 is CCZ with both target controls negated.
    ccz=(('H',(2,)),)+ccx+(('H',(2,)),)
    J_inverse=(('CX',(1,2)),('Ry',-2*theta,(1,)))
    J=(('Ry',2*theta,(1,)),('CX',(1,2)))
    flips=(('X',(1,)),('X',(2,)))
    return J_inverse+flips+ccz+flips+J+((('Z',(0,)),) if negative else ())


def six_arbitrary_matrices(theta):
    """Six unrestricted two-wire gates, in time order, with explicit matrices."""
    X=gates.make_gate('X');XX=sp.kronecker_product(X,X)
    J=gates.make_gate('CX')*sp.kronecker_product(sp.eye(2),gates.make_gate('Ry',2*theta))
    CS=gates.make_gate('CS');CSdg=gates.make_gate('CSdg')
    return ((CS*XX*J.H,(1,2)),(gates.make_gate('CX'),(0,1)),
            (CSdg,(1,2)),(gates.make_gate('CX'),(0,1)),
            (CS,(0,2)),(J*XX,(1,2)))


def reflection_schmidt_minor(theta):
    """Exact OSR determinant; nonzero throughout 0<theta<=pi/4."""
    if __package__:
        from .symbolic_osr import realign
    else:
        from symbolic_osr import realign
    return sp.factor(realign(reflection(theta),(0,)).det())


def singlet_frame():
    return sp.kronecker_product(gates.make_gate('X'),gates.make_gate('Z'))
