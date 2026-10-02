/* Link this standalone test against a QGD_DFE-enabled libqgd.
 * Run with the mock libqgdDFE.so on LD_LIBRARY_PATH:
 *   dfe_backend_compatibility fpga
 *   dfe_backend_compatibility groq
 *   dfe_backend_compatibility fpga /path/to/groq/mock/libqgdDFE.so
 * The last case verifies that globally loaded Groq symbols do not identify an
 * unrelated FPGA plugin as Groq. No accelerator hardware is needed.
 */
#include "Gates_block.h"
#include "Optimization_Interface.h"
#include "U3.h"
#include "CNOT.h"
#include "CRY.h"
#include <algorithm>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static int32_t fixed(double value) {
    return static_cast<int32_t>(value * (1 << 25));
}

static DFEgate_kernel_type u3(int target, const double *p, bool groq) {
    return {fixed(groq ? p[0] : p[0]/2), fixed(p[1]), fixed(p[2]),
            static_cast<int8_t>(target), -1, U3_OPERATION, 0};
}

static void check_records(bool groq) {
    static_assert(sizeof(DFEgate_kernel_type) == 16, "DFE ABI changed");
    Gates_block circuit(5);
    circuit.add_gate(new U3(5, 0));
    circuit.add_gate(new CNOT(5, 4, 3));
    Gates_block *nested = new Gates_block(5);
    nested->add_gate(new U3(5, 2));
    nested->add_gate(new CRY(5, 1, 0));
    circuit.add_gate(nested);
    circuit.add_gate(new U3(5, 4));

    Matrix_real parameters(1, 10);
    for (int i = 0; i < 10; ++i) parameters[i] = .11 * (i + 1);
    const double *p = parameters.get_data();
    std::vector<DFEgate_kernel_type> expected = {
        u3(0, p, groq), {fixed(M_PI/2), 0, fixed(M_PI), 4, 3, CNOT_OPERATION, 0},
        u3(2, p + 3, groq), {fixed(p[6]), 0, 0, 1, 0, CRY_OPERATION, 0},
        u3(4, p + 7, groq),
    };
    if (!groq) std::reverse(expected.begin(), expected.end());

    int gates = 0, sets = 0, redundant = 0;
    std::unique_ptr<DFEgate_kernel_type[]> actual(
        circuit.convert_to_DFE_gates_with_derivates(parameters, gates, sets, redundant));
    check(gates == 5 && sets == 12 && redundant == 1, "unexpected gate padding");
    for (int i = 0; i < gates; ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(expected[i])) != 0) {
            std::cerr << "gate " << i << ": actual " << actual[i].ThetaOver2 << ','
                      << actual[i].Phi << ',' << actual[i].Lambda << ','
                      << int(actual[i].target_qbit) << ',' << int(actual[i].control_qbit)
                      << "; expected " << expected[i].ThetaOver2 << ',' << expected[i].Phi
                      << ',' << expected[i].Lambda << ',' << int(expected[i].target_qbit)
                      << ',' << int(expected[i].control_qbit) << std::endl;
        }
        check(std::memcmp(&actual[i], &expected[i], sizeof(expected[i])) == 0,
              "base gate order, angles or ABI differs");
    }

    const int forward_position[] = {0, 0, 0, 2, 2, 2, 3, 4, 4, 4};
    const int component[] = {0, 1, 2, 0, 1, 2, 0, 0, 1, 2};
    for (int parameter = 0; parameter < 10; ++parameter) {
        std::vector<DFEgate_kernel_type> derivative = expected;
        int position = forward_position[parameter];
        if (!groq) position = gates - position - 1;
        DFEgate_kernel_type& gate = derivative[position];
        if (component[parameter] == 0) {
            gate.ThetaOver2 += fixed(M_PI/2);
            gate.metadata = 128;
        } else if (component[parameter] == 1) {
            gate.Phi += fixed(M_PI/2);
            gate.metadata = 128 + 3;
        } else {
            gate.Lambda += fixed(M_PI/2);
            gate.metadata = 128 + 5;
        }
        for (int i = 0; i < gates; ++i) {
            check(std::memcmp(&actual[(parameter + 1)*gates + i], &derivative[i],
                              sizeof(derivative[i])) == 0,
                  "derivative gate order, parameter index or metadata differs");
        }
    }
}

static void check_dispatch(bool groq) {
    void *mock = dlopen(DFE_LIB_9QUBITS, RTLD_NOW);
    check(mock != NULL, "mock plugin not found");
    int (*call_count)() = reinterpret_cast<int (*)()>(dlsym(mock, "dfe_test_call_count"));
    check(call_count != NULL, "not a test plugin");
    for (int qubits : {2, 4, 5}) {
        Matrix identity(1 << qubits, 1 << qubits);
        for (int i = 0; i < identity.rows; ++i) {
            for (int j = 0; j < identity.cols; ++j) {
                identity[i*identity.cols + j].real = i == j ? 1 : 0;
                identity[i*identity.cols + j].imag = 0;
            }
        }
        std::map<std::string, Config_Element> config;
        Optimization_Interface model(identity, qubits, false, config, ZEROS, 1);
        model.add_gate(new U3(qubits, 0));
        model.upload_Umtx_to_DFE();
        Matrix_real parameters(1, 3), gradient(1, 3);
        parameters[0] = .2; parameters[1] = .3; parameters[2] = .4;
        const int before = call_count();
        double value = 0;
        model.optimization_problem_combined(parameters, &value, gradient);
        check(call_count() - before == (groq || qubits >= 5 ? 1 : 0),
              "combined objective dispatch threshold differs");
    }
    dlclose(mock);
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 || argc == 3, "expected fpga/groq and optional global mock");
        const bool groq = std::string(argv[1]) == "groq";
        void *unrelated = argc == 3 ? dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL) : NULL;
        check(argc != 3 || unrelated != NULL, "global mock not found");
        check(!is_groq_dfe(), "backend should default to FPGA before initialization");
        check(init_dfe_lib(1, 5, 1) == 1, "mock initialization failed");
        check(is_groq_dfe() == groq, "wrong backend detection");
        check_records(groq);
        unload_dfe_lib();
        check(!is_groq_dfe(), "backend flag not reset on unload");
        check_dispatch(groq);
        unload_dfe_lib();
        check(!is_groq_dfe(), "backend flag not reset after reinitialization");
        if (unrelated) dlclose(unrelated);
        std::cout << "PASS " << (groq ? "Groq" : "FPGA")
                  << " records, nested derivatives, dispatch and unload" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (const std::string& error) {
        std::cerr << error << std::endl;
        return 1;
    }
}
