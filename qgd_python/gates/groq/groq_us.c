#include <groq/groqio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
//#define NUM_DEVICE 1
Device device[NUM_DEVICE] = {NULL};
IOP prog = NULL;
IOBufferArray* inputBuffers[NUM_DEVICE] = {NULL};
IOBufferArray* outputBuffers[NUM_DEVICE] = {NULL};
TensorLayout inpLayouts[6] = {NULL};
TensorLayout outpLayouts[2] = {NULL};
unsigned int loaded = 0;

#define releive_groq releive_DFE 
#define initialize_groq initialize_DFE 
#define load2groq load2LMEM
#define calcqgdKernelGroq_oneShot calcqgdKernelDFE_oneShot
#define calcqgdKernelGroq calcqgdKernelDFE

extern "C" void releive_groq()
{
    if (!loaded) return;
    Status status;
    if (prog) {
        unsigned int num_programs = 0;
        status = groq_iop_get_number_of_programs(prog, &num_programs);
        
        for (size_t i = 0; i < NUM_DEVICE; i++) {
            for (unsigned int j = 0; j < num_programs; j++) {  
                if (inputBuffers[i] && inputBuffers[i][j]) {
                    status = groq_deallocate_iobuffer_array(inputBuffers[i][j]);
                    if (status) {
                        printf("ERROR: groq deallocate iobuffer error %d\n", status);
                    }
                }
                if (outputBuffers[i] && outputBuffers[i][j]) {
                    status = groq_deallocate_iobuffer_array(outputBuffers[i][j]);
                    if (status) {
                        printf("ERROR: groq deallocate iobuffer error %d\n", status);
                    }
                }
            }
            if (inputBuffers[i]) {
                free(inputBuffers[i]);
                inputBuffers[i] = NULL;
            }
            if (outputBuffers[i]) {            
                free(outputBuffers[i]);
                outputBuffers[i] = NULL;
            }
        }
    
        status = groq_iop_deinit(prog);
        if (status) {
            printf("ERROR: IOP deinit error %d\n", status);
        }
        prog = NULL;
    }

    for (size_t i = 0; i < NUM_DEVICE; i++) {
        if (device[i]) {
            status = groq_device_close(device[i]);
            if (status) {
                printf("ERROR: close device error %d\n", status);
            }
            device[i] = NULL;
        }
    }

    if (driver) {
        status = groq_deinit(&driver);
        if (status) {
            printf("ERROR: close device error %d\n", status);
        }
    
        driver = NULL;
    }
    loaded = 0;
}

const int MAX_LEVELS = 6;

//xxd -i us2-20.0.iop us2-20.0.iop.h
#ifdef NUM_QBITS
#if NUM_QBITS==2
#include "us2-20.0.iop.h"
#define binary_groq_program usiop_us2_20_0_iop
#elif NUM_QBITS==3
#include "us3-57.0.iop.h"
#define binary_groq_program usiop_us3_57_0_iop
#elif NUM_QBITS==4
#include "us4-112.0.iop.h"
#define binary_groq_program usiop_us4_112_0_iop
#elif NUM_QBITS==5
#include "us5-185.0.iop.h"
#define binary_groq_program usiop_us5_185_0_iop
#elif NUM_QBITS==6
#include "us6-276.0.iop.h"
#define binary_groq_program usiop_us6_276_0_iop
#elif NUM_QBITS==7
#include "us7-385.0.iop.h"
#define binary_groq_program usiop_us7_385_0_iop
#elif NUM_QBITS==8
#include "us8-512.0.iop.h"
#define binary_groq_program usiop_us8_512_0_iop
#elif NUM_QBITS==9
#include "us9-657.0.iop.h"
#define binary_groq_program usiop_us9_657_0_iop
#elif NUM_QBITS==10
#include "us10-820.0.iop.h"
#define binary_groq_program usiop_us10_820_0_iop
#endif
#define MAX_GATES (NUM_QBITS+3*(NUM_QBITS*(NUM_QBITS-1)/2*MAX_LEVELS))
#else
#include "us2-20.0.iop.h"
#include "us3-57.0.iop.h"
#include "us4-112.0.iop.h"
#include "us5-185.0.iop.h"
#include "us6-276.0.iop.h"
#include "us7-385.0.iop.h"
#include "us8-512.0.iop.h"
#include "us9-657.0.iop.h"
#include "us10-820.0.iop.h"
unsigned char* binary_groq_programs[] = {
    usiop_us2_20_0_iop, usiop_us3_57_0_iop, usiop_us4_112_0_iop,
    usiop_us5_185_0_iop, usiop_us6_276_0_iop, usiop_us7_385_0_iop,
    usiop_us8_512_0_iop, usiop_us9_657_0_iop, usiop_us10_820_0_iop
};
size_t prog_sizes[] = {
    sizeof(usiop_us2_20_0_iop), sizeof(usiop_us3_57_0_iop), sizeof(usiop_us4_112_0_iop),
    sizeof(usiop_us5_185_0_iop), sizeof(usiop_us6_276_0_iop), sizeof(usiop_us7_385_0_iop),
    sizeof(usiop_us8_512_0_iop), sizeof(usiop_us9_657_0_iop), sizeof(usiop_us10_820_0_iop)
};
#define MAX_GATES(NUM_QBITS) (NUM_QBITS+3*(NUM_QBITS*(NUM_QBITS-1)/2*MAX_LEVELS))
size_t max_gates[] = {
    MAX_GATES(2), MAX_GATES(3), MAX_GATES(4), MAX_GATES(5), MAX_GATES(6),
    MAX_GATES(7), MAX_GATES(8), MAX_GATES(9), MAX_GATES(10)
};
#endif

const char* TENSOR_TYPES[] = {"UNKNOWN", "UINT8", "UINT16", "UINT32", "INT8", "INT16", "INT32", "FLOAT16", "FLOAT32", "BOOL"};

#ifdef NUM_QBITS
int initialize_groq()
#else
int initialize_groq(unsigned int num_qbits)
#endif
{
#ifdef NUM_QBITS
  if (loaded) return 0;
#else
  if (num_qbits == loaded) return 0;
  else if (loaded) releive_groq();
#endif

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
#ifdef NUM_QBITS
  size_t prog_size = sizeof(binary_groq_program);
#else
  size_t prog_size = prog_sizes[num_qbits-2];
#endif
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

      //printf("device %d %p, %s, %d, %d\n",
      //        i, device[devidx], bus_addr, curr_proc_id, numa_node);
      /***** END DEVICE OBJECT TESTING *****/
      devidx++;
    }
    if (devidx != NUM_DEVICE) {
        printf("ERROR: number of requested devices not acquired\n");
        return 1;
    }
#ifdef NUM_QBITS
    status = groq_iop_init(binary_groq_program, prog_size, &prog);
#else
    status = groq_iop_init(binary_groq_programs[num_qbits-2], prog_size, &prog);
#endif
    if (status) {
        printf("ERROR: IOP init %d\n", status);
        return 1;
    }
    //printf("prog %p\n", prog);
    unsigned int num_programs = 0;
    status = groq_iop_get_number_of_programs(prog, &num_programs);
    if (status) {
      printf("ERROR: iop get number of programs error %d\n", status);
      return 1;
    }
    for (unsigned int d = 0; d < NUM_DEVICE; ++d) {
      //printf("number of programs: %u\n", num_programs);
      inputBuffers[d] = (IOBufferArray*)malloc(sizeof(IOBufferArray)*num_programs);
      outputBuffers[d] = (IOBufferArray*)malloc(sizeof(IOBufferArray)*num_programs);

      for (unsigned int j = 0; j < num_programs; j++) {
        status = groq_load_program(device[d], prog, j, true);
        if (status) {
            printf("ERROR: load program %d\n", status);
            return 1;
        }

        //printf("loaded program: %u\n", j);
        char* name = NULL;
        status = groq_program_name(prog, j, &name);
        if (status) {
          printf("ERROR: program name error %d\n", status);
          return 1;
        }
        //printf("program %u name: %s\n", j, name);
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
        size_t inpSize, outpSize;
        uint32_t ptindex;
        //for (size_t i = 0; i < n; i++) {
        for (size_t i = 0; i < 1; i++) {//only need the first entrypoint, there are 4 all with identical inputs/outputs
            EntryPoint ep;
            status = groq_get_nth_entrypoint(p, 0, &ep);
            if (status) {
                printf("ERROR: get nth entrypoint %d\n", status);
                return 1;
            }
            status = groq_entrypoint_get_ptindex(ep, &ptindex);
            if (status) {
                printf("ERROR: entrypoint get ptindex %d\n", status);
                return 1;
            }
            //printf("entrypoint %lu ptindex %u\n", i, ptindex);
            IODescriptor iod;
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
                size_t l;
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
                    if (strcmp(name, "unitaryinit") == 0) {
                        memcpy(&inpLayouts[0], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "gate") == 0) {
                        memcpy(&inpLayouts[1], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "othergate") == 0) {
                        memcpy(&inpLayouts[2], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "target_qbits") == 0) {
                        memcpy(&inpLayouts[3], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "control_qbits") == 0) {
                        memcpy(&inpLayouts[4], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "derivates") == 0) {
                        memcpy(&inpLayouts[5], &tl, sizeof(TensorLayout));
                    } else if (strcmp(name, "singledim") == 0) {
                        memcpy(&outpLayouts[j-(num_programs-2)], &tl, sizeof(TensorLayout));
                    } else {
                        printf("Unknown tensor: %s\n", name);
                        return 1;
                    }
                    /*uint32_t type;
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
                    }*/
                }
            }
        }
        inpSize = groq_program_get_input_size(prog, j);
        outpSize = groq_program_get_output_size(prog, j);
        //printf("input size %lu output size %lu\n", inpSize, outpSize);
        //program table index entry point offsets: 0 is (input, output, compute), 1 is (input only), 2 is (compute only), 3 is (output only)
        status = groq_allocate_iobuffer_array(driver, inpSize, 1, ptindex+((j<=1||j>=num_programs-2) ? 0 : 2), &inputBuffers[d][j]);
        if (status) {
            printf("ERROR: allocate iobuffer array %d\n", status);
            return 1;
        }                                            
        status = groq_allocate_iobuffer_array(driver, outpSize, 1, ptindex+((j<=1||j>=num_programs-2) ? 0 : 2), &outputBuffers[d][j]);
        if (status) {
            printf("ERROR: allocate iobuffer array %d\n", status);
            return 1;
        }
      }
  }
#ifdef NUM_QBITS
  loaded = NUM_QBITS;
  printf("Initialized Groq for %d qbits on %d devices\n", NUM_QBITS, NUM_DEVICE);
#else  
  loaded = num_qbits;
  printf("Initialized Groq for %d qbits on %d devices\n", num_qbits, NUM_DEVICE);
#endif
  return 0;
}

extern "C" int get_chained_gates_num() {
    return 1;
}


unsigned char* floatToBytes(float* f) //endianness dependent method
{
    return (unsigned char*)f;
}

float* bytesToFloat(unsigned char* c) //endianness dependent method
{
    return (float*)c;
}

void* curdata = NULL;

//https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
unsigned int log2(unsigned int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;
    
    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
                                            r |= (v >> 1);
    return r;
}

#define USE_GROQ_HOST_FUNCS

extern "C" int load2groq(Complex8* data, size_t rows, size_t cols)
{
    // test whether the DFE engine can be initialized
#ifdef NUM_QBITS
    //size_t mx_gates = MAX_GATES;
    if ( initialize_groq() ) { //cols must equal rows and be a power of 2 from 4 to 1024 for 2 to 10 qbits
#else
    int num_qbits = log2(cols);
    //size_t mx_gates = max_gates[num_qbits-2];
    if ( initialize_groq(num_qbits) ) {
#endif
        printf("Failed to initialize the Groq engine\n");
        releive_groq();
        return 1;
    }
    //data type e.g. FLOAT32 along the -2 dimension
    //input format:
    //gate FLOAT32 ((MAX_GATES+1)/2, 8, gatecols)
    //othergate FLOAT32 ((MAX_GATES+1)/2, 8, gatecols)
    //target_qbits UINT8 (MAX_GATES, 320)
    //control_qbits UINT8 (MAX_GATES, 320)
    //unitary FLOAT32 (2*rows, cols)
    //Python source order is unitary, gate, othergate, target_qbits, control_qbits
    //however the host order is address sorted so by hemisphere, slice then offset EASTERNMOST to WESTERNMOST initialization order so E43->E0 then W0->W43, little endian matching CPU endianness
    //making order othergate, gate, unitary, target_qbits, control_qbits, derivates
    u_int8_t* input;
    size_t num_inner_splits = (cols+320-1)/320;
#ifndef USE_GROQ_HOST_FUNCS
    size_t offset = 0; //(mx_gates+1)/2*2*8*4*320;
    size_t maxinnerdim = cols > 320 ? 256 : cols; //real chip shape is (num_inner_splits, rows, 2, min(cols, 256)) as inner splits are 256 for 9 and 10 qbits (not for 11)
    size_t imag_offset = offset+4*rowinner;
    size_t rowinner = num_inner_splits*rows*320;
#endif
    Completion completion[NUM_DEVICE];
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        Status status = groq_get_data_handle(inputBuffers[d][0], 0, &input);
        if (status) {
            printf("ERROR: get data handle %d\n", status);
            return 1;
        }
#ifdef USE_GROQ_HOST_FUNCS
        float* buf = (float*)malloc(sizeof(float)*2*rows*cols);
        for (size_t i = 0; i < rows; i++) {
            size_t offs = i*cols;
            size_t offs2 = offs*2; 
            for (size_t j = 0; j < cols; j++) {
                buf[offs2+j] = data[offs+j].real;
                buf[offs2+cols+j] = data[offs+j].imag;
            }
        }
        status = groq_tensor_layout_from_host(inpLayouts[0], (unsigned char*)buf, sizeof(float)*2*rows*cols, input, num_inner_splits*2*rows*4*320);
        if (status) {
            printf("ERROR: tensor layout from host %d\n", status);
            return 1;
        }
        free(buf);
#else
        for (size_t i = 0; i < rows; i++) { //effective address order dimension as S8 is (2, 4, num_inner_splits, rows, 320)
            for (size_t j = 0; j < cols; j++) {
                unsigned char* re = floatToBytes(&data[i*cols+j].real), *im = floatToBytes(&data[i*cols+j].imag);
                size_t innerdim = j % maxinnerdim;
                size_t innersplit = j / maxinnerdim;
                size_t inneroffs = innersplit*rows*320+i*320+innerdim;
                for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                    //if (input[offset+byteidx*num_inner_splits*rows*320+innersplit*rows*320+i*320+innerdim] != re[byteidx] ||
                    //    input[offset+4*num_inner_splits*rows*320+byteidx*num_inner_splits*rows*320+innersplit*rows*320+i*320+innerdim] != im[byteidx])
                    //    printf("bad unitary data format %lu %lu %lu\n", i, j, byteidx);
                    input[offset+byteidx*rowinner+inneroffs] = re[byteidx];
                    input[imag_offset+byteidx*rowinner+inneroffs] = im[byteidx];
                }
            }
        }
#endif
        status = groq_invoke(device[d], inputBuffers[d][0], 0, outputBuffers[d][0], 0, &completion[d]);
        if (status) {
            printf("ERROR: invoke %d\n", status);
            return 1;
        }
    }
    for (size_t d = 0; d < NUM_DEVICE; d++) {
        int completionCode;
        while (!(completionCode = groq_poll_completion(completion[d])));
        if (completionCode != GROQ_COMPLETION_SUCCESS) {
            printf("ERROR: %s\n", completionCode == GROQ_COMPLETION_GFAULT ? "GFAULT" : "DMA FAULT");
            return 1;
        }
    }
    return 0;
}

int calcqgdKernelGroq_oneShot(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace )
{
    //printf("oneShot with %d gates and %d gate sets\n", gatesNum, gateSetNum);
    int curGateSet[NUM_DEVICE];
    for (int d = 0; d < NUM_DEVICE; d++) { curGateSet[d] = -1; }
    int curStep[NUM_DEVICE] = {0};
    int nextGateSet = 0;
    Completion completion[NUM_DEVICE];
#ifdef NUM_QBITS
    int num_qbits = NUM_QBITS;
    size_t mx_gates = MAX_GATES;
#else
    int num_qbits = log2(cols);
    size_t mx_gates = max_gates[num_qbits-2];
#endif
    //size_t num_inner_splits = (cols+320-1)/320;
    size_t mx_gates320 = mx_gates*320;
    size_t hemigates = (mx_gates+1)/2;
#ifndef USE_GROQ_HOST_FUNCS
    size_t hemigates16 = hemigates*2*320;
    size_t offset_qbit = hemigates*2*8*4*320; //num_inner_splits*2*rows*4*320
#endif
    size_t gatecols = cols < 320 ? cols : 320;
    size_t maxinnerdim = cols > 320 ? 256 : cols; //real chip shape is (num_inner_splits, rows, 2, min(cols, 256)) as inner splits are 256 for 9 and 10 qbits (not for 11)
    size_t hemigatessz = hemigates*8*gatecols*sizeof(float);
    size_t gateinputsz = hemigates*2*8*4*320+mx_gates*3*320;
    Status status;
    if (gateSetNum == 0) return 0;
    while (true) {
        for (int d = 0; d < NUM_DEVICE; d++) {        
            if (curGateSet[d] == -1) {
                if (nextGateSet >= gateSetNum) continue;
                curGateSet[d] = nextGateSet++;
                curStep[d] = 0;
                u_int8_t* input;
                status = groq_get_data_handle(inputBuffers[d][1], 0, &input);
                if (status) {
                    printf("ERROR: get data handle %d\n", status);
                    return 1;
                }
#ifdef USE_GROQ_HOST_FUNCS
                float* gbuf1 = (float*)calloc(hemigates*8*gatecols, sizeof(float));
                float* gbuf2 = (float*)calloc(hemigates*8*gatecols, sizeof(float));
                unsigned char* tbuf = (unsigned char*)malloc(mx_gates320);
                unsigned char* cbuf = (unsigned char*)malloc(mx_gates320);
                unsigned char* dbuf = (unsigned char*)malloc(mx_gates320);
                for (int i = 0; i < gatesNum; i++) {
                    int idx = i+curGateSet[d]*gatesNum;
                    size_t ioffs = i/2*8*gatecols;
                    long double c = cosl(gates[idx].Theta), s = sinl(gates[idx].Theta);
                    float _Complex g[] = { (gates[idx].metadata & 1) ? 0.0 : c,
                        (gates[idx].metadata & 2) ? 0.0 : -cexpl(I*gates[idx].Lambda)*s,
                        (gates[idx].metadata & 4) ? 0.0 : cexpl(I*gates[idx].Phi)*s,
                        (gates[idx].metadata & 8) ? 0.0 : cexpl(I*(gates[idx].Phi+gates[idx].Lambda))*c };
                    for (size_t goffs = 0; goffs < 4; goffs++) {
                        size_t baseoffs = ioffs+goffs*2*gatecols,
                            baseoffsimag = ioffs+(goffs*2+1)*gatecols;
                        /*for (size_t innerdim = 0; innerdim < gatecols; innerdim++) {
                            if ((i & 1) == 0) {
                                gbuf1[baseoffs+innerdim] = crealf(g[goffs]);
                                gbuf1[baseoffsimag+innerdim] = cimagf(g[goffs]);
                            } else {
                                gbuf2[baseoffs+innerdim] = crealf(g[goffs]);
                                gbuf2[baseoffsimag+innerdim] = cimagf(g[goffs]);
                            }
                        }*/
                        float re = crealf(g[goffs]), im = cimagf(g[goffs]);
                        wmemset((wchar_t*)((i & 1) == 0 ? &gbuf1[baseoffs] : &gbuf2[baseoffs]), *((wchar_t*)floatToBytes(&re)), gatecols);
                        wmemset((wchar_t*)((i & 1) == 0 ? &gbuf1[baseoffsimag] : &gbuf2[baseoffsimag]), *((wchar_t*)floatToBytes(&im)), gatecols);
                    }
                    for (size_t j = 0; j < 320; j++) tbuf[320*i+j] = (j & 15) <= 1 ? gates[idx].target_qbit%8*2 + (j & 15) : 16;
                    for (size_t j = 0; j < 320; j++) cbuf[320*i+j] = (j & 15) <= 1 ? (gates[idx].control_qbit - (gates[idx].control_qbit > gates[idx].target_qbit ? 1 : 0))%8*2 + (j & 15) : 16;
                    int deriv = (gates[idx].metadata & 0x80) != 0;
                    memset(&dbuf[320*i], deriv, 320);
                }
                status = groq_tensor_layout_from_host(inpLayouts[1], (unsigned char*)gbuf1, hemigatessz, input, gateinputsz);
                if (status) {
                    printf("ERROR: tensor layout from host %d\n", status);
                    return 1;
                }
                status = groq_tensor_layout_from_host(inpLayouts[2], (unsigned char*)gbuf2, hemigatessz, input, gateinputsz);
                if (status) {
                    printf("ERROR: tensor layout from host %d\n", status);
                    return 1;
                }
                status = groq_tensor_layout_from_host(inpLayouts[3], tbuf, mx_gates320, input, gateinputsz);
                if (status) {
                    printf("ERROR: tensor layout from host %d\n", status);
                    return 1;
                }
                status = groq_tensor_layout_from_host(inpLayouts[4], cbuf, mx_gates320, input, gateinputsz);
                if (status) {
                    printf("ERROR: tensor layout from host %d\n", status);
                    return 1;
                }
                status = groq_tensor_layout_from_host(inpLayouts[5], dbuf, mx_gates320, input, gateinputsz);
                if (status) {
                    printf("ERROR: tensor layout from host %d\n", status);
                    return 1;
                }
                free(gbuf1); free(gbuf2); free(tbuf); free(cbuf); free(dbuf);
#else                
                for (int i = 0; i < gatesNum; i++) {
                    int idx = i+curGateSet[d]*gatesNum;
                    size_t ioffs = i/2*2*320;
                    gate_kernel_type* curgate = &gates[idx];
                    long double c = cosl(curgate->Theta), s = sinl(curgate->Theta);
                    float _Complex g[] = { (curgate->metadata & 1) != 0 ? 0.0 : c,
                        (curgate->metadata & 2) != 0 ? 0.0 : -cexpl(I*curgate->Lambda)*s,
                        (curgate->metadata & 4) != 0 ? 0.0 : cexpl(I*curgate->Phi)*s,
                        (curgate->metadata & 8) != 0 ? 0.0 : cexpl(I*(curgate->Phi+curgate->Lambda))*c };
                    size_t offset = (i & 1) != 0 ? hemigates*8*4*320 : 0;
                    for (size_t goffs = 0; goffs < 4; goffs++) { //effective address order dimension as S16 is (4, 4, hemigates, 2, min(320, cols))
                        size_t gioffs = offset+ioffs+goffs/2*320;
                        size_t goffsre = goffs%2*2*4, goffsim = (goffs%2*2+1)*4;
                        float fr = crealf(g[goffs]), fi = cimagf(g[goffs]);
                        //printf("%f+%fj ", fr, fi);
                        //printf("%X+%Xj ", *((unsigned int*)floatToBytes(&fr)), *((unsigned int*)floatToBytes(&fi)));  
                        unsigned char* re = floatToBytes(&fr), *im = floatToBytes(&fi);
                        for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                            size_t baseoffs = (((i & 1) == 0 ? 15-(goffsre+byteidx) : goffsre+byteidx))*hemigates16+gioffs;
                            size_t baseoffsimag = (((i & 1) == 0 ? 15-(goffsim+byteidx) : goffsim+byteidx))*hemigates16+gioffs;
                            memset(&input[baseoffs], re[byteidx], gatecols);
                            memset(&input[baseoffsimag], im[byteidx], gatecols);
                        }
                    }
                    //printf("\n");
                    offset = offset_qbit + i * 320;
                    for (size_t j = 0; j < 320; j++) {
                        //if (input[offset+j] != ((j & 15) <= 1 ? (curgate->control_qbit - (curgate->control_qbit > curgate->target_qbit ? 1 : 0))%8*2 + (j & 15) : 16))
                        //    printf("bad control qbit format\n"); 
                        input[offset+j] = (j & 15) <= 1 ? (curgate->control_qbit - (curgate->control_qbit > curgate->target_qbit ? 1 : 0))%8*2 + (j & 15) : 16;
                    }
                    offset += mx_gates * 320;
                    for (size_t j = 0; j < 320; j++) {
                        //if (input[offset+j] != ((j & 15) <= 1 ? curgate->target_qbit%8*2 + (j & 15) : 16))
                        //    printf("bad target qbit format\n");
                        input[offset+j] = (j & 15) <= 1 ? curgate->target_qbit%8*2 + (j & 15) : 16;
                    }
                    offset += mx_gates * 320;
                    int deriv = (curgate->metadata & 0x80) != 0;
                    memset(&input[offset], deriv, 320);
                }
#endif
                status = groq_invoke(device[d], inputBuffers[d][1], 0, outputBuffers[d][1], 0, &completion[d]);
                if (status) {
                    printf("ERROR: invoke %d\n", status);
                    return 1;
                }
            } else {
                int completionCode=groq_poll_completion(completion[d]);
                if (completionCode) {
                    if (completionCode != GROQ_COMPLETION_SUCCESS) {
                        printf("ERROR: %s\n", completionCode == GROQ_COMPLETION_GFAULT ? "GFAULT" : "DMA FAULT");
                        return 1;
                    }
                    if (curStep[d] == gatesNum+1) {
                        int progidx = 1+1+(2+(num_qbits >= 9 ? 2 : 0)+(num_qbits >= 10 ? 2 : 0))*2+(gatesNum&1);
                        u_int8_t *output;
                        status = groq_get_data_handle(outputBuffers[d][progidx], 0, &output);
                        if (status) {
                            printf("ERROR: get data handle %d\n", status);
                            return 1;
                        }        
                        //output format when unitary returned: FLOAT32 (2*rows, cols) where inner splits are outermost dimension
                        double curtrace = 0.0;
/*#ifdef USE_GROQ_HOST_FUNCS
                        Complex8* data = (Complex8*)malloc(sizeof(Complex8)*rows*cols);
                        float* buf = (float*)malloc(2*rows*cols*sizeof(float)); //logical shape of result is (num_inner_splits, rows, 2, min(256, cols))
                        status = groq_tensor_layout_to_host(outpLayouts[gatesNum&1], output, num_inner_splits*2*rows*4*320, (unsigned char*)buf, 2*rows*cols*sizeof(float)); 
                        if (status) {
                            printf("ERROR: tensor layout to host %d\n", status);
                            return 1;
                        }
                        free(buf);
                        for (size_t i = 0; i < rows; i++) {
                            size_t j = i;
                            size_t innerdim = j % maxinnerdim;
                            size_t innersplit = j / maxinnerdim;
                            curtrace += buf[innersplit*rows*2*maxinnerdim+i*2*maxinnerdim+innerdim];
                        }
#else
                        for (size_t i = 0; i < rows; i++) {
                            size_t j = i;
                            //for (size_t j = 0; j < cols; j++) {
                                size_t innerdim = j % maxinnerdim;
                                size_t innersplit = j / maxinnerdim;
                                unsigned char rb[sizeof(float)]; //, ib[sizeof(float)];
                                //unsigned char* vrb = floatToBytes(&buf[innersplit*rows*2*maxinnerdim+i*2*maxinnerdim+innerdim]), *vib = floatToBytes(&buf[innersplit*rows*2*maxinnerdim+i*2*maxinnerdim+maxinnerdim+innerdim]);
                                for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                                    rb[byteidx] = output[((gatesNum & 1) != 0 ? 4+3-byteidx : byteidx)*num_inner_splits*rows*320+innersplit*rows*320+i*320+innerdim]; 
                                    //ib[byteidx] = output[((gatesNum & 1) != 0 ? 3-byteidx : 4+byteidx)*num_inner_splits*rows*320+innersplit*rows*320+i*320+innerdim];
                                    //if (vrb[byteidx] != rb[byteidx] || vib[byteidx] != ib[byteidx]) printf("bad output format %lu %lu %lu\n", i, j, byteidx);
                                }
                                float re = *bytesToFloat(rb); //, im = *bytesToFloat(ib);
                                //data[i*cols+j].real = re;
                                //data[i*cols+j].imag = im;
                                //printf("%f+%fj ", re, im);
                                //printf("%X+%Xj ", *((unsigned int*)floatToBytes(&re)), *((unsigned int*)floatToBytes(&im)));
                                if (i == j) curtrace += re;
                            //}
                            //printf("\n");
                        }
#endif*/
#ifdef USE_GROQ_HOST_FUNCS
                        float* buf = (float*)malloc(maxinnerdim*sizeof(float));
                        status = groq_tensor_layout_to_host(outpLayouts[gatesNum&1], output, 320*sizeof(float), (unsigned char*)buf, maxinnerdim*sizeof(float));
                        if (status) {
                            printf("ERROR: tensor layout to host %d\n", status);
                            return 1;
                        } 
                        for (size_t i = 0; i < maxinnerdim; i++) {
                            curtrace += buf[i];
                        }
                        free(buf);
#else
                        for (size_t i = 0; i < maxinnerdim; i++) {                            
                            unsigned char rb[sizeof(float)]; //hemisphere placement WEST on both output programs so no order reversal
                            for (size_t byteidx = 0; byteidx < sizeof(float); byteidx++) {
                                rb[byteidx] = output[320*byteidx+i];
                            }
                            float re = *bytesToFloat(rb);
                            curtrace += re;                            
                        }
#endif
                        //printf("\n");
                        trace[curGateSet[d]] = curtrace;
#ifdef TEST
                        printf("%f\n", curtrace);
#endif
                        //free(data);
                        
                        curGateSet[d] = -1;
                    } else if (curStep[d] == gatesNum) {
                        int progidx = 1+1+(2+(num_qbits >= 9 ? 2 : 0)+(num_qbits >= 10 ? 2 : 0))*2+(gatesNum&1);
                        status = groq_invoke(device[d], inputBuffers[d][progidx], 0, outputBuffers[d][progidx], 0, &completion[d]);
                        if (status) {
                            printf("ERROR: invoke %d\n", status);
                            return 1;
                        }
                        curStep[d]++;
                    } else {
                        int idx = curStep[d]+curGateSet[d]*gatesNum;
                        int progidx = 1+((curStep[d]&1)!=0 ? 2+(num_qbits >= 9 ? 2 : 0)+(num_qbits >= 10 ? 2 : 0) : 0) + gates[idx].target_qbit/8*2 + (gates[idx].target_qbit == gates[idx].control_qbit ? 0 : 1+(2+(gates[idx].target_qbit/8==0))*((gates[idx].control_qbit-(gates[idx].control_qbit > gates[idx].target_qbit ? 1 : 0))/8));
                        status = groq_invoke(device[d], inputBuffers[d][progidx], 0, outputBuffers[d][progidx], 0, &completion[d]);
                        if (status) {
                            printf("ERROR: invoke %d\n", status);
                            return 1;
                        }
                        curStep[d]++;
                    }
                }
            }
        }
        if (nextGateSet >= gateSetNum) {
            int d;
            for (d = 0; d < NUM_DEVICE; d++) {
                if (curGateSet[d] != -1) break; 
            }
            if (d == NUM_DEVICE) break;
        }        
    }    
    return 0;
}

extern "C" int calcqgdKernelGroq(size_t rows, size_t cols, gate_kernel_type* gates, int gatesNum, int gateSetNum, double* trace)
{
    /*for (int gateSet = 0; gateSet < gateSetNum; gateSet += NUM_DEVICE) {
        if (calcqgdKernelGroq_oneShot(rows, cols, gates+gateSet, gatesNum, gateSetNum-gateSet < NUM_DEVICE ? gateSetNum-gateSet : NUM_DEVICE, trace+gateSet)) return 1;
    }
    return 0;*/
    return calcqgdKernelGroq_oneShot(rows, cols, gates, gatesNum, gateSetNum, trace);
}

#ifdef TEST
int main(int argc, char* argv[])
{
    if (initialize_groq()) exit(1);
    int rows = 1 << NUM_QBITS, cols = 1 << NUM_QBITS;
    Complex8* data = (Complex8*)calloc(rows*cols, sizeof(Complex8));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            //data[i*cols+j].real = i*cols*2+j;
            //data[i*cols+j].imag = i*cols*2+cols+j;
            if (i==j) data[i*cols+j].real = 1.0;
        }
    }
    load2groq(data, rows, cols);
    int gatesNum = MAX_GATES,
        gateSetNum = 4;
    gate_kernel_type* gates = (gate_kernel_type*)calloc(gatesNum * gateSetNum, sizeof(gate_kernel_type));
    for (int i = 0; i < gatesNum; i++) {
        for (int d = 0; d < gateSetNum; d++) {
            gates[i+d*gatesNum].Theta = 25+i+d;
            gates[i+d*gatesNum].Lambda = 50+i;
            gates[i+d*gatesNum].Phi = 55+i;
            gates[i+d*gatesNum].target_qbit = i % NUM_QBITS;
            gates[i+d*gatesNum].control_qbit = i % NUM_QBITS;
            gates[i+d*gatesNum].metadata = 0;
        } 
    }
    double* trace = (double*)calloc(gateSetNum, sizeof(double));
    calcqgdKernelGroq(rows, cols, gates, gatesNum, gateSetNum, trace);
    free(data);
    free(gates);
    releive_groq();
    return 0;
}
#endif
