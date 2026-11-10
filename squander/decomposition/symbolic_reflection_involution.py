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


def seven_cnot_source(theta, negative=False):
    """Literal seven-CX word; only two U3 polar angles depend on theta.

    Wire order is c=0, a=1, b=2, as in make_controlled. All constants are
    rational multiples of pi; the fixed scalar is absorbed in physical U3s.
    The old eight-CX constructor remains for archive provenance.
    """
    p = sp.pi
    word = (
        ('U3', p, 0, 0, (0,)), ('U3', p, -p/2, 3*p/4, (0,)),
        ('U3', -2*p, p/4, -3*p/4, (1,)), ('U3', -2*p, p/4, -p/4, (2,)),
        ('CX', (0, 1)),
        ('U3', -2*p, p/4, -p/2, (0,)), ('U3', p, -p/4, -p/2, (1,)),
        ('CX', (1, 2)),
        ('U3', p-2*theta, -3*p/4, -3*p/4, (1,)), ('U3', p, -p/4, 0, (2,)),
        ('CX', (0, 1)),
        ('U3', p, -p/2, -3*p/4, (0,)), ('U3', 0, 3*p/4, -p/2, (1,)),
        ('CX', (1, 2)),
        ('U3', p, p/4, -p/4, (1,)), ('U3', p, 0, 3*p/4, (2,)),
        ('CX', (0, 2)),
        ('U3', p, 3*p/4, p/4, (0,)), ('U3', 0, p/2, -p/4, (2,)),
        ('CX', (1, 2)),
        ('U3', 2*p-2*theta, -p, 0, (1,)), ('U3', p, 3*p/4, -p/2, (2,)),
        ('CX', (1, 2)),
        ('U3', p, -p/4, 3*p/4, (1,)), ('U3', p, 0, 0, (2,)),
    )
    return word + ((('Z', (0,)),) if negative else ())


def six_arbitrary_matrices(theta):
    """Six unrestricted two-wire gates, in time order, with explicit matrices."""
    X=gates.make_gate('X');XX=sp.kronecker_product(X,X)
    J=gates.make_gate('CX')*sp.kronecker_product(sp.eye(2),gates.make_gate('Ry',2*theta))
    CS=gates.make_gate('CS');CSdg=gates.make_gate('CSdg')
    return ((CS*XX*J.H,(1,2)),(gates.make_gate('CX'),(0,1)),
            (CSdg,(1,2)),(gates.make_gate('CX'),(0,1)),
            (CS,(0,2)),(J*XX,(1,2)))


def five_arbitrary_matrices(theta):
    """Five unrestricted gates, exactly, for every real Schmidt angle.

    With X on the first target, B R(theta) X B.H = Z tensor S
    (little-endian). B is an explicitly unitary eigenframe; no eigenvalue
    fitting or numerical local equivalence is used.
    """
    r=sp.sqrt(2)/2
    B0=r*sp.Matrix([[0,0,1,1],[0,0,-1,1],[-sp.I,1,0,0],[sp.I,1,0,0]])
    J=gates.apply_to(sp.eye(4),2,gates.make_gate('Ry',2*theta),(1,))
    J=gates.apply_to(J,2,gates.make_gate('CX'),(1,0))
    B=sp.ImmutableMatrix(B0*J.H)
    return ((gates.make_gate('CX'),(0,1)),(B,(1,2)),
            (gates.make_gate('CS'),(0,2)),(gates.make_gate('CZ'),(0,1)),(B.H,(1,2)))


def reflection_schmidt_minor(theta):
    """Exact OSR determinant; nonzero throughout 0<theta<=pi/4."""
    if __package__:
        from .symbolic_osr import realign
    else:
        from symbolic_osr import realign
    return sp.factor(realign(reflection(theta),(0,)).det())


def singlet_frame():
    return sp.kronecker_product(gates.make_gate('X'),gates.make_gate('Z'))
