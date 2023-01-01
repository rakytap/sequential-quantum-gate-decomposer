#include <groq/groqio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tgmath.h>
#include <complex.h>

typedef struct {
  float real;
  float imag;
} Complex8;

typedef struct {
	int32_t Theta;
	int32_t Phi;
	int32_t Lambda;
	int8_t target_qbit;
	int8_t control_qbit;
	int8_t gate_type;
	uint8_t metadata;
} gate_kernel_type;

Driver driver = NULL;
#define NUM_DEVICE 2
Device device[NUM_DEVICE] = {NULL};
IOP prog = NULL;
IOBufferArray* inputBuffers[NUM_DEVICE] = {NULL};
IOBufferArray* outputBuffers[NUM_DEVICE] = {NULL};

#define releive_groq releive_DFE 
#define initialize_groq initialize_DFE 
#define load2groq load2LMEM
#define calcqgdKernelGroq_oneShot calcqgdKernelDFE_oneShot
#define calcqgdKernelGroq calcqgdKernelDFE

void releive_groq()
{
    unsigned int num_programs = 0;
    Status status = groq_iop_get_number_of_programs(prog, &num_programs);
    for (size_t i = 0; i < NUM_DEVICE; i++) {
        for (unsigned int j = 0; j < num_programs; j++) {  
            status = groq_deallocate_iobuffer_array(inputBuffers[i][j]);
            if (status) {
                printf("ERROR: groq deallocate iobuffer error %d\n", status);
            }
            status = groq_deallocate_iobuffer_array(outputBuffers[i][j]);
            if (status) {
                printf("ERROR: groq deallocate iobuffer error %d\n", status);
            }
        }
        free(inputBuffers[i]);
        free(outputBuffers[i]);
    }

    status = groq_iop_deinit(prog);
    if (status) {
        printf("ERROR: IOP deinit error %d\n", status);
    }

    for (size_t i = 0; i < NUM_DEVICE; i++) {
        status = groq_device_close(device[i]);
        if (status) {
            printf("ERROR: close device error %d\n", status);
        }
    }

    status = groq_deinit(&driver);
    if (status) {
        printf("ERROR: close device error %d\n", status);
    }

    driver = NULL;
}

//xxd -i us2-20.0.iop us2-20.0.iop.h
#if NUM_QBITS==2
#include "us2-20.0.iop.h"
#define binary_groq_program us2_20_0_iop
#elif NUM_QBITS==3
#include "us3-57.0.iop.h"
#define binary_groq_program us3_57_0_iop
#elif NUM_QBITS==4
#include "us4-112.0.iop.h"
#define binary_groq_program us4_112_0_iop
#elif NUM_QBITS==5
#include "us5-185.0.iop.h"
#define binary_groq_program us5_185_0_iop
#elif NUM_QBITS==6
#include "us6-276.0.iop.h"
#define binary_groq_program us6_276_0_iop
#elif NUM_QBITS==7
#include "us7-385.0.iop.h"
#define binary_groq_program us7_385_0_iop
#elif NUM_QBITS==8
#include "us8-512.0.iop.h"
#define binary_groq_program us8_512_0_iop
#elif NUM_QBITS==9
#include "us9-657.0.iop.h"
#define binary_groq_program us9_657_0_iop
#elif NUM_QBITS==10
#include "us10-820.0.iop.h"
#define binary_groq_program us10_820_0_iop
#endif

const int MAX_LEVELS = 6;
#define MAX_GATES (NUM_QBITS+3*(NUM_QBITS*(NUM_QBITS-1)/2*MAX_LEVELS))

const char* TENSOR_TYPES[] = {"UNKNOWN", "UINT8", "UINT16", "UINT32", "INT8", "INT16", "INT32", "FLOAT16", "FLOAT32", "BOOL"};

int initialize_groq()
{
  Status status = groq_init(&driver);
  if (status) {
    printf("ERROR: groq init error %d\n", status);
    return 1;
  }

  unsigned int num_devices = 0;
  status = groq_get_number_of_devices(driver, &num_devices);
  if (status) {
    printf("ERROR: get num of devices error %d\n", status);
    return 1;
  }
  if (num_devices > NUM_DEVICE) num_devices = NUM_DEVICE;
  // Find the size of the file and allocate corresponding buffer
  size_t prog_size = sizeof(binary_groq_program);
  size_t devidx = 0;
  for (unsigned int i = 0; i < num_devices; ++i) {
      /***** START DEVICE OBJECT TESTING *****/
      //status = groq_get_next_available_device(driver, &device);
      status = groq_get_nth_device(driver, i, &device[devidx]);
      if (status) {
        printf("ERROR: get nth device error %d, i = %d\n", status, i);
        return 1;
      }

      char const *bus_addr = NULL;
      status = groq_device_bus_address(device[devidx], &bus_addr);
      if (status) {
        printf("ERROR: device bus address error %d\n", status);
        return 1;
      }

      status = groq_device_open(device[devidx]);
      if (status){
        printf("ERROR: device %d open error %d\n", i, status);
        continue; //return 1;
      }

      if(!groq_device_is_locked(device[devidx])) {
        printf("ERROR: device is not locked\n");
        return 1;
      }

      unsigned char const *ecid = NULL;
      status = groq_device_ecid(device[devidx], &ecid);
      if (status) {
        printf("ERROR: device ecid error %d\n", status);
        return 1;
      }

      unsigned long csr_value = 0;
      status = groq_device_read_csr(device[devidx], 0x18, &csr_value);
      if (status) {
        printf("ERROR: device read csr error %d\n", status);
        return 1;
      }

      unsigned int dcr_value = 0;
      status = groq_device_read_dcr(device[devidx], 0x18, &dcr_value);
      if (status) {
        printf("ERROR: device read dcr error %d\n", status);
        return 1;
      }

      int curr_proc_id = 0;
      status = groq_device_curr_proc_id(device[devidx], &curr_proc_id);
      if (status) {
        printf("ERROR: device curr proc id error %d\n", status);
        return 1;
      }

      int numa_node = 0;
      status = groq_device_numa_node(device[devidx], &numa_node);
      if (status) {
        printf("ERROR: device numa node error %d\n", status);
        return 1;
      }

      printf("device %d %p, %s, %d, %d\n",
              i, device[devidx], bus_addr, curr_proc_id, numa_node);
      /***** END DEVICE OBJECT TESTING *****/

    }
    status = groq_iop_init(binary_groq_program, prog_size, &prog);
    if (status) {
        printf("ERROR: IOP init %d\n", status);
        return 1;
    }
    printf("prog %p\n", prog);
    unsigned int num_programs = 0;
    status = groq_iop_get_number_of_programs(prog, &num_programs);
    if (status) {
      printf("ERROR: iop get number of programs error %d\n", status);
      return 1;
    }
    for (unsigned int i = 0; i < NUM_DEVICE; ++i) {
      printf("number of programs: %u\n", num_programs);
      inputBuffers[i] = (IOBufferArray*)malloc(sizeof(IOBufferArray)*num_programs);
      outputBuffers[i] = (IOBufferArray*)malloc(sizeof(IOBufferArray)*num_programs);

      for (unsigned int j = 0; j < num_programs; j++) {
        status = groq_load_program(device[i], prog, j, true);
        if (status) {
            printf("ERROR: load program %d\n", status);
            return 1;
        }    
                  
        printf("loaded program: %u\n", j);
        char* name = NULL;
        status = groq_program_name(prog, j, &name);
        if (status) {
          printf("ERROR: program name error %d\n", status);
          return 1;
        }
        printf("program %u name: %s\n", j, name);
        Program p;
        status = groq_get_nth_program(prog, j, &p);
        if (status) {
            printf("ERROR: get nth program %d\n", status);
            return 1;
        }            
        size_t n;
        /*status = groq_program_get_metadata_count(p, &n);
        if (status) {
            printf("ERROR: program get metadata count %d\n", status);
            return 1;
        }            
        for (size_t i = 0; i < n; i++) {
            const char* key, *value;
            status = groq_program_get_nth_metadata(p, i, &key, &value);
            if (status) {
                printf("ERROR: program get nth metadata %d\n", status);
                return 1;
            }            
        }*/
        status = groq_get_number_of_entrypoints(p, &n);
        if (status) {
            printf("ERROR: program get number of entrypoints %d\n", status);
            return 1;
        }            
        //for (size_t i = 0; i < n; i++) {
        for (size_t i = 0; i < 1; i++) {//only need the first entrypoint, there are 4 all with identical inputs/outputs
            EntryPoint ep;
            status = groq_get_nth_entrypoint(p, 0, &ep);
            if (status) {
                printf("ERROR: get nth entrypoint %d\n", status);
                return 1;
            }            
            uint32_t ptindex;
            status = groq_entrypoint_get_ptindex(ep, &ptindex);
            if (status) {
                printf("ERROR: entrypoint get ptindex %d\n", status);
                return 1;
            }                        
            printf("entrypoint %lu ptindex %u\n", i, ptindex);
            IODescriptor iod;
            size_t inpSize, outpSize;
            for (size_t inpoutp = 0; inpoutp < 2; inpoutp++) {
                if (inpoutp == 0) {
                    status = groq_entrypoint_get_input_iodescriptor(ep, &iod);
                    if (status) {
                        printf("ERROR: entrypoint get input iodescriptor %d\n", status);
                        return 1;
                    }                                            
                    status = groq_iodescriptor_get_size(iod, &inpSize);
                    if (status) {
                        printf("ERROR: iodescriptor get size %d\n", status);
                        return 1;
                    }                                            
                } else {
                    status = groq_entrypoint_get_output_iodescriptor(ep, &iod);
                    if (status) {
                        printf("ERROR: entrypoint get output iodescriptor %d\n", status);
                        return 1;
                    }                                            
                    status = groq_iodescriptor_get_size(iod, &outpSize);
                    if (status) {
                        printf("ERROR: iodescriptor get size %d\n", status);
                        return 1;
                    }                                            
                }
                /*size_t l;
                status = groq_iodescriptor_get_number_of_tensor_layouts(iod, &l);
                if (status) {
                    printf("ERROR: iodescriptor get number of tensor layouts %d\n", status);
                    return 1;
                }                                            
                for (size_t k = 0; k < l; k++) {
                    TensorLayout tl;
                    status = groq_iodescriptor_get_nth_tensor_layout(iod, k, &tl);
                    if (status) {
                        printf("ERROR: iodescriptor get nth tensor layout %d\n", status);
                        return 1;
                    }                                                                
                    char* name;
                    status = groq_tensor_layout_get_name(tl, &name);
                    if (status) {
                        printf("ERROR: tensor layout get name %d\n", status);
                        return 1;
                    }                                                                
                    uint32_t type;
                    status = groq_tensor_layout_get_type(tl, &type);
                    if (status) {
                        printf("ERROR: tensor layout get type %d\n", status);
                        return 1;
                    }                                                                
                    size_t ndim;
                    status = groq_tensor_layout_get_number_of_dimensions(tl, &ndim);
                    if (status) {
                        printf("ERROR: tensor layout get number of dimensions %d\n", status);
                        return 1;
                    }                                                                
                    int32_t format;
                    status = groq_tensor_layout_get_format(tl, &format);
                    if (status) {
                        printf("ERROR: tensor layout get format %d\n", status);
                        return 1;
                    }                                                                
                    size_t size;
                    status = groq_tensor_layout_get_size(tl, &size);
                    if (status) {
                        printf("ERROR: tensor layout get size %d\n", status);
                        return 1;
                    }                                                                
                    printf("tensor %lu %s type %u %s ndim %lu format %d size %lu (", k, name, type, TENSOR_TYPES[type], ndim, format, size);   
                    for (size_t m = 0; m < ndim; m++) {
                        uint32_t dimSize;
                        status = groq_tensor_layout_get_nth_dimension(tl, m, &dimSize);
                        if (status) {
                            printf("ERROR: tensor layout get nth dimension %d\n", status);
                            return 1;
                        }                                                                
                        printf(m == ndim-1 ? "%u)\n" : "%u, ", dimSize); 
                    }
                }*/
            }
            printf("input size %lu output size %lu\n", inpSize, outpSize);
            status = groq_allocate_iobuffer_array(driver, inpSize, 1, ptindex, &inputBuffers[i][j]);
            if (status) {
                printf("ERROR: allocate iobuffer array %d\n", status);
                return 1;
            }                                            
            status = groq_allocate_iobuffer_array(driver, outpSize, 1, ptindex, &outputBuffers[i][j]);
            if (status) {
                printf("ERROR: allocate iobuffer array %d\n", status);
                return 1;
            }
        }
      }
  }
  return 0;
}

char* floatToBytes(float* f) //endianness dependent method
{
    return (char*)f;
}

float* bytesToFloat(char* c) //endianness dependent method
{
    return (float*)c;
}

void* curdata = NULL;

int load2groq(Complex8* data, size_t rows, size_t cols)
{
    //data type e.g. FLOAT32 along the -2 dimension
    //input format:
    //gate FLOAT32 ((MAX_GATES+1)/2, 8, min(320, cols))
    //othergate FLOAT32 ((MAX_GATES+1)/2, 8, min(320, cols))
    //target_qbits UINT8 (MAX_GATES, 320)
    //control_qbits UINT8 (MAX_GATES, 320)
    //unitary FLOAT32 (2*rows, cols) 
    u_int8_t* input;
    size_t offset = (MAX_GATES+1)/2*2*8*4*320 + MAX_GATES * 320 * 2;
    size_t maxinnerdim = cols > 320 ? 256 : cols; //real chip shape is (num_inner_splits, rows, 2, min(cols, 256)) as inner splits are 256 for 9 and 10 qbits (not for 11)
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        Status status = groq_get_data_handle(inputBuffers[d][0], 0, &input);
        if (status) {
            printf("ERROR: get data handle %d\n", status);
            return 1;
        }
        //int num_inner_splits = (rows+320-1)/320;
        for (size_t i = 0; i < rows; i++) { 
            for (size_t j = 0; j < cols; j++) {
                //inputs[tensornames["unitary"]] = np.ascontiguousarray(u.astype(np.complex64)).view(np.float32).reshape(pow2qb, pow2qb, 2).transpose(0, 2, 1).reshape(pow2qb*2, pow2qb)
                char* re = floatToBytes(&data[i*cols+j].real), *im = floatToBytes(&data[i*cols+j].imag);
                size_t innerdim = j % maxinnerdim;
                size_t innersplit = j / maxinnerdim;
                for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                    input[offset+innersplit*rows*2*320+i*2*320+byteidx*320+innerdim] = re[byteidx];
                    input[offset+innersplit*rows*2*320+(i*2+1)*320+byteidx*320+innerdim] = im[byteidx];
                }
            }
        }
    }
    return 0;
}

int calcqgdKernelGroq_oneShot(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
{
    size_t offset_qbit = (MAX_GATES+1)/2*2*8*4*320;
    size_t gatecols = cols < 320 ? cols : 320;
    Status status;
    for (int d = 0; d < gateSetNum; d++) {
        u_int8_t* input;
        status = groq_get_data_handle(inputBuffers[d][0], 0, &input);
        if (status) {
            printf("ERROR: get data handle %d\n", status);
            return 1;
        }        
        for (int i = 0; i < gatesNum; i++) {
            long double c = cosl(gates[i+d*gatesNum].Theta), s = sinl(gates[i+d*gatesNum].Theta);
            float _Complex g[] = { c, -cexpl(I*gates[i+d*gatesNum].Lambda)*s, cexp(I*gates[i+d*gatesNum].Phi)*s, cexpl(I*(gates[i+d*gatesNum].Phi+gates[i+d*gatesNum].Lambda))*c };
            size_t offset = ((i & 1) != 0 ? (MAX_GATES+1)/2*8*4*320 : 0) + (i / 2)*4*320;
            for (size_t goffs = 0; goffs < 4; goffs++) {
                float fr = crealf(g[goffs]), fi = cimagf(g[goffs]);
                char* re = floatToBytes(&fr), *im = floatToBytes(&fi);
                for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                    for (size_t innerdim = 0; innerdim < gatecols; innerdim++) {
                        input[offset+goffs*2*4*320+byteidx*320+innerdim] = re[byteidx];
                        input[offset+(goffs*2+1)*4*320+byteidx*320+innerdim] = im[byteidx];
                    }
                }
            }
            offset = offset_qbit + i * 320;
            for (size_t j = 0; j < 320; j++) input[offset+j] = (j & 15) <= 1 ? gates[i+d*gatesNum].target_qbit%8*2 + (j & 15) : 16;
            offset += MAX_GATES * 320;
            for (size_t j = 0; j < 320; j++) input[offset+j] = (j & 15) <= 1 ? (gates[i+d*gatesNum].control_qbit - (gates[i+d*gatesNum].control_qbit > gates[i+d*gatesNum].target_qbit ? 1 : 0))%8*2 + (j & 15) : 16;
        }
    }
    Completion completion[NUM_DEVICE];
    unsigned long completionCode;
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        status = groq_invoke(device[d], inputBuffers[0], 0, outputBuffers[0], 0, &completion[d]);
        if (status) {
            printf("ERROR: invoke %d\n", status);
            return 1;
        }
    }
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        while (!(completionCode=groq_poll_completion(completion[d])));
        if (completionCode != GROQ_COMPLETION_SUCCESS) {
            printf("ERROR: %s\n", completionCode == GROQ_COMPLETION_GFAULT ? "GFAULT" : "DMA FAULT");
            return 1;
        }
    }
    
    int progidx;
    for (int i = 0; i < gatesNum; i++) {
        for (size_t d = 0; d < NUM_DEVICE; d++) {
            progidx = 1+((i&1)!=0 ? 2+(NUM_QBITS >= 9 ? 2 : 0)+(NUM_QBITS >= 10 ? 2 : 0) : 0) + gates[i+d*gatesNum].target_qbit/8*2 + (gates[i+d*gatesNum].target_qbit == gates[i+d*gatesNum].control_qbit ? 0 : 1+(2+(gates[i+d*gatesNum].target_qbit/8==0))*((gates[i+d*gatesNum].control_qbit-(gates[i+d*gatesNum].control_qbit > gates[i+d*gatesNum].target_qbit ? 1 : 0))/8));
            printf("%lu %d %d\n", d, i, progidx);
            status = groq_invoke(device[d], inputBuffers[d][progidx], 0, outputBuffers[d][progidx], 0, &completion[d]);
            if (status) {
                printf("ERROR: invoke %d\n", status);
                return 1;
            }
        }
        for (size_t d = 0; d < NUM_DEVICE; d++) {
            while (!(completionCode=groq_poll_completion(completion[d])));
            if (completionCode != GROQ_COMPLETION_SUCCESS) {
                printf("ERROR: %s\n", completionCode == GROQ_COMPLETION_GFAULT ? "GFAULT" : "DMA FAULT");
                return 1;
            }
        }
    }
    progidx = 1+(2+(NUM_QBITS >= 9 ? 2 : 0)+(NUM_QBITS >= 10 ? 2 : 0))*2+(gatesNum&1);
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        status = groq_invoke(device[d], inputBuffers[d][progidx], 0, outputBuffers[d][progidx], 0, &completion[d]);
        if (status) {
            printf("ERROR: invoke %d\n", status);
            return 1;
        }
    }
    for (size_t d = 0; d < NUM_DEVICE; d++) {    
        while (!(completionCode=groq_poll_completion(completion[d])));
        if (completionCode != GROQ_COMPLETION_SUCCESS) {
            printf("ERROR: %s\n", completionCode == GROQ_COMPLETION_GFAULT ? "GFAULT" : "DMA FAULT");
            return 1;
        }
    }
    size_t maxinnerdim = cols > 320 ? 256 : cols; //real chip shape is (num_inner_splits, rows, 2, min(cols, 256)) as inner splits are 256 for 9 and 10 qbits (not for 11)
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        u_int8_t *output;
        status = groq_get_data_handle(outputBuffers[d][progidx], 0, &output);
        if (status) {
            printf("ERROR: get data handle %d\n", status);
            return 1;
        }    
        //output format if unitary returned: FLOAT32 (2*rows, cols) where inner splits are outermost dimension
        Complex8* data = (Complex8*)malloc(sizeof(Complex8)*rows*cols);
        double curtrace = 0.0;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                //result[0] = np.ascontiguousarray(res[0][tensornames["unitaryres" if (num_gates&1)==0 else "unitaryrevres"]].reshape(num_inner_splits, pow2qb, 2, min(256, pow2qb)).transpose(1, 0, 3, 2)).view(np.complex64).reshape(pow2qb, pow2qb).astype(np.complex128)
                size_t innerdim = j % maxinnerdim;
                size_t innersplit = j / maxinnerdim;
                char rb[sizeof(float)], ib[sizeof(float)];
                for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                    rb[byteidx] = output[innersplit*rows*2*320+i*2*320+byteidx*320+innerdim]; 
                    ib[byteidx] = output[innersplit*rows*2*320+(i*2+1)*320+byteidx*320+innerdim];
                }
                float re = *bytesToFloat(rb), im = *bytesToFloat(ib);
                data[i*cols+j].real = re;
                data[i*cols+j].imag = im;
                if (i == j) curtrace += sqrt((double)re * (double)re + (double)im * (double)im);
            }
        }
        trace[d] = curtrace;
        free(data);
    }
    return 0;
}

int calcqgdKernelGroq(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace)
{
    for (int gateSet = 0; gateSet < gateSetNum; gateSet += NUM_DEVICE) {
        if (calcqgdKernelGroq_oneShot(rows, cols, gates+gateSet, gatesNum, gateSetNum-gateSet < NUM_DEVICE ? gateSetNum-gateSet : NUM_DEVICE, trace+gateSet)) return 1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    if (initialize_groq()) exit(1);
    int num_qbits = 2;
    int rows = 1 << num_qbits, cols = 1 << num_qbits;
    Complex8* data = (Complex8*)malloc(sizeof(Complex8)*rows*cols);
    load2groq(data, rows, cols);
    int gatesNum = MAX_GATES, gateSetNum = 4;
    gate_kernel_type* gates = (gate_kernel_type*)calloc(sizeof(gate_kernel_type), gatesNum);
    double trace = 0.0;
    calcqgdKernelGroq(rows, cols, gates, gatesNum, gateSetNum, &trace);
    free(data);
    free(gates);
    releive_groq();
    return 0;
}
