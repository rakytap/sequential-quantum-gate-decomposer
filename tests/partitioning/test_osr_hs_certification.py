"""The OSR HS certificate must survive fixed-structure export refinement."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from squander import N_Qubit_Decomposition_Tree_Search
from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit


# Matrix-only reproduction of a reduced three-wire synthesis target. Neither
# its known gate word nor source angles are supplied to the synthesizer.
_INVERSE = np.asarray([
    [(-0.4466293231453982-0.10774322694629969j), (-0.15401048495486294-0.06162381743418704j), (-0.3240659676477836+0.027351462904464007j), (0.1576108328130367-0.2453746150007732j), (0.3222323999947827+0.3702056319004571j), (0.0783369289624105+0.1768541059375915j), (0.3497276533177553+0.25805459455064444j), (-0.3016921550512644+0.11117755968991107j)],
    [(0.13554441918942758-0.023363814654335055j), (0.42298041781768037-0.1535426795600004j), (-0.013058161097365692+0.3048919954832213j), (0.256076430993153-0.21970772306490155j), (-0.1330710677559064-0.09659126762640412j), (-0.46329417201475237-0.12596893467713807j), (0.21882208717877105-0.25766366895642057j), (-0.4469280495394325-0.007210390959906274j)],
    [(0.05298648121687522-0.12832714360272163j), (-0.39405325632383026+0.21437346661744716j), (0.27688100152670336+0.2189331338779686j), (0.370340519605067-0.08092397368003607j), (-0.17514888639276643-0.0408255545802079j), (0.47150320030411713+0.08904179066527344j), (-0.0721967016060481-0.2812755535385241j), (-0.3534471467588544-0.19837912171890895j)],
    [(0.21026776132977193-0.40500571063882995j), (0.012908397394073765+0.16160755350315392j), (-0.24330765500448387+0.2735544783360923j), (-0.3355864147932175-0.048025430205946164j), (-0.4544487657579744+0.18890912286259923j), (0.19710446223134834-0.07389396658344666j), (0.39154762041093355-0.04080331725510177j), (0.18972473437472057+0.19755534457551277j)],
    [(-0.24869956046728442+0.28162368451890335j), (-0.13791048258584107-0.0756394276132142j), (-0.5748039397536451+0.12835488951359014j), (-0.0538835441495848-0.23430367423916607j), (-0.31494288479512933+0.11033910690497756j), (-0.0337324429611774-0.11706957902612831j), (-0.44657640969369133-0.25359290877444374j), (0.062468769097458104-0.18860197875133328j)],
    [(0.1537466482500298+0.010921386578351974j), (0.04330073808721397-0.3718939384501106j), (0.14819059780488333+0.19324417007125622j), (0.3996297866864182-0.434364702598523j), (0.07988381200583912+0.09620878513980739j), (0.19710871795074697-0.270751329120725j), (0.01912905949720347+0.1944066462306335j), (0.51024071643739-0.04561881058843766j)],
    [(0.5301315919175836+0.17202299384197967j), (-0.09266135864695355+0.23614827459807053j), (-0.14189102674909213-0.2911825786758446j), (0.15383788364922224-0.08732139907702173j), (0.24040298578930408+0.4896625795995196j), (-0.13801064486875197+0.12765412571680335j), (0.08093479474617664-0.3723390449033293j), (0.1034457417229931+0.0034048337891618478j)],
    [(-0.018441289059873127-0.25804211162220897j), (-0.5352422408413123+0.1634650920204255j), (-0.09833692069098046+0.14734045934864257j), (0.2841709550607394+0.15574166316595445j), (0.07497604844651302-0.16533709355402496j), (-0.4736673555960817-0.26611561794654737j), (-0.09713692003195121+0.031620680325002974j), (0.1427684241702089+0.3528835738651548j)],
], dtype=np.complex128)
_TOLERANCE = 3.4360835511554885e-11
_CONFIG = {'max_inner_iterations_bfgs2': 1000, 'max_iteration_loops': 8, 'osr_certified_cnot_deflation': True, 'osr_certified_terminal_completion': True, 'osr_cnot_deflation_basin_hops': 1, 'osr_cut_smoothmax_temperature': 0.1, 'osr_eager_projection_restarts': True, 'osr_geometric_completion_basin_hops': 2, 'osr_geometric_completion_motifs': True, 'osr_optimization_tolerance': 1e-06, 'osr_path_deterministic_rng': True, 'osr_physical_cover_motifs': True, 'osr_profile_temperature': 0.1, 'osr_rank_constructive_pair_class': False, 'osr_rank_degenerate_cover_class': True, 'parallel': 0, 'random_seed': 20260926, 'use_basin_hopping': True, 'use_differential_evolution': False, 'use_dual_annealing': False, 'use_float': False, 'use_graph_search': True, 'use_osr': True, 'optimization_tolerance': 3.4360835511554885e-11, 'synthesis_acceptance_tolerance': 3.4360835511554885e-11, 'max_partition_size': 3, 'tree_level_max': 4}


@pytest.mark.parametrize("parent_cost", [0, 3])
def test_osr_certification_uses_hs_in_search_and_export(parent_cost):
    solver = N_Qubit_Decomposition_Tree_Search(
        _INVERSE.copy(), config=dict(_CONFIG), accelerator_num=0,
        topology=[(0, 1), (0, 2), (1, 2)],
    )
    solver.set_Verbose(0)
    solver.set_Cost_Function_Variant(parent_cost)
    solver.set_Optimizer("BFGS2")
    solver.set_Optimization_Tolerance(1e-14)
    solver.Start_Decomposition()
    circuit = solver.get_Circuit()
    parameters = np.asarray(solver.get_Optimized_Parameters(), dtype=np.float64)
    exported = get_Qiskit_Circuit(circuit, parameters)
    native = np.asarray(circuit.get_Matrix(parameters))
    independent = Operator(exported).data
    target = _INVERSE.conj().T
    for matrix in (native, independent):
        overlap = np.vdot(target, matrix)
        phase = overlap / abs(overlap)
        residual = matrix - phase * target
        stable_error = float(np.vdot(residual, residual).real / target.shape[0])
        assert stable_error < _TOLERANCE
        hs_error = 1.0 - abs(overlap) ** 2 / target.shape[0] ** 2
        assert hs_error < _TOLERANCE
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(8), atol=1e-12, rtol=0)
    assert exported.count_ops().get("cx", 0) <= 4
    assert solver.get_Decomposition_Error() < _TOLERANCE
