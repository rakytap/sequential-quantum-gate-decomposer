"""Basis selection for exact symbolic synthesis with one entangler family.

The presets describe mathematical gate sets, not a claim that the legacy
heuristic search already certifies minimum entangler counts in each basis.
Continuous presets are universal up to global phase; retain an explicit global
phase separately when exact matrix equality is required. Discrete dense bases
cannot exactly represent an arbitrary symbolic rotation angle.
"""

from dataclasses import dataclass

if __package__:
    from .symbolic_gates import gate_definition
else:
    from symbolic_gates import gate_definition


@dataclass(frozen=True)
class BasisSet:
    name: str
    single_qubit_gates: tuple
    entangler: str
    universality: str
    notes: str = ""

    @property
    def gates(self):
        return (self.entangler,) + self.single_qubit_gates

    @property
    def exact_arbitrary_angles(self):
        return self.universality == "exact_continuous"

    def entangler_count(self, sequence):
        """Count only this basis's primitive two-qubit operations.

        Entries use the existing (name, *parameters, wires) convention. This is
        a cost measurement, not a certificate that the sequence is optimal.
        """
        normalize = lambda name: {"U3": "U", "U1": "P", "CNOT": "CX"}.get(name, name)
        allowed = {normalize(name) for name in self.gates}
        count = 0
        for gate in sequence:
            name = normalize(gate_definition(gate[0]).name)
            if name not in allowed:
                raise ValueError(f"Gate {gate[0]!r} is not in basis {self.name!r}")
            count += name == normalize(self.entangler)
        return count


def make_basis(name, single_qubit_gates, entangler, *, universality="unspecified", notes=""):
    """Validate one-entangler syntax; custom universality is a caller assertion."""
    local = tuple(dict.fromkeys(gate_definition(g).name for g in single_qubit_gates))
    if not local or any(gate_definition(g).num_qubits != 1 for g in local):
        raise ValueError("Single-qubit basis entries must all have exactly one qubit")
    entangler = gate_definition(entangler).name
    if gate_definition(entangler).num_qubits != 2:
        raise ValueError("Choose one two-qubit entangler or parameterized entangler family")
    if entangler in {"SWAP", "XX", "YY", "ZZ"}:
        raise ValueError(f"{entangler} alone does not supply an entangling primitive")
    if universality not in {"exact_continuous", "approximate_discrete", "clifford", "unspecified"}:
        raise ValueError("Unknown universality classification")
    return BasisSet(name, local, entangler, universality, notes)


BASIS_SETS = {}


def _preset(name, local, entangler, mode="exact_continuous", notes=""):
    BASIS_SETS[name] = make_basis(name, local, entangler, universality=mode, notes=notes)


_preset("cx_u", ("U3",), "CX")
_preset("cx_rz_ry", ("Rz", "Ry"), "CX")
_preset("cx_h_phase", ("H", "P"), "CX")
_preset("ibm_cx", ("Rz", "Sx", "X"), "CX")
_preset("ibm_ecr", ("Rz", "Sx", "X"), "ECR")
_preset("ibm_cz", ("Rz", "Sx", "X"), "CZ")
_preset("cz_u", ("U3",), "CZ")
_preset("cz_rz_rx", ("Rz", "Rx"), "CZ")
_preset("iswap_u", ("U3",), "iSWAP")
_preset("sqrt_iswap_u", ("U3",), "SiSWAP")
_preset("sqrt_swap_u", ("U3",), "SSWAP")
_preset("xx90_u", ("U3",), "XX90")
_preset("xx_rz_ry", ("Rz", "Ry"), "Rxx")
_preset("zz_u", ("U3",), "Rzz")
_preset("zx_u", ("U3",), "Rzx")
_preset("google_cz", ("PhasedXZ",), "CZ")
_preset("google_sqrt_iswap", ("PhasedXZ",), "SiSWAP")
_preset("google_syc", ("PhasedXZ",), "SYC")
_preset("google_fsim", ("PhasedXZ",), "fSim")
_preset("berkeley_b", ("U3",), "B")
_preset("ionq_aria", ("GPi", "GPi2"), "IonQMS",
        notes="Native turns; interaction theta restricted to [0,1/4] on hardware.")
_preset("ionq_aria_full", ("GPi", "GPi2"), "IonQMSMax")
_preset("ionq_forte", ("GPi", "GPi2"), "IonQZZ")
_preset("quantinuum_zzmax", ("PhasedX", "Rz"), "ZZMax")
_preset("quantinuum_zzphase", ("PhasedX", "Rz"), "ZZPhase")
_preset("tket_tk2", ("PhasedX", "Rz"), "TK2",
        notes="Compiler macro count; expand TK2 before comparing physical pulse counts.")
_preset("rigetti_cz", ("Rz", "Sx", "X"), "CZ",
        notes="Sx differs from native Rx(pi/2) by a tracked global phase.")
_preset("aqt_rxx", ("R", "Rz"), "Rxx")
_preset("iqm_cz", ("R",), "CZ",
        notes="R(theta,phi) represents PRX in radians; IQM API units require conversion.")
_preset("clifford_t_cx", ("H", "S", "T", "Tdg"), "CX", "approximate_discrete",
        "Dense universal; exact only for representable targets, not arbitrary symbolic angles.")
_preset("clifford_t_cz", ("H", "S", "T", "Tdg"), "CZ", "approximate_discrete")
_preset("clifford_only", ("H", "S"), "CX", "clifford", "Clifford analysis/control; not universal.")


def get_basis(name):
    try:
        return BASIS_SETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown basis {name!r}; choose from {', '.join(BASIS_SETS)}") from error
