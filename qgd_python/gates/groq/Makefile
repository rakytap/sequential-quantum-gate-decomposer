GROQ_DIR ?= /opt/groq/runtime

CC = g++

# define any compile-time flags
CFLAGS = -Wall -g -fPIC

# define any directories containing header files other than /usr/include
INCLUDES = -I$(GROQ_DIR)/include

# define library paths in addition to /usr/lib
LFLAGS = 

# define any libraries to link into executable:
LIBS = $(GROQ_DIR)/lib/groqruntime/lib/libgroqruntime.a -lrt

# define the C source files
SRCS = groq_us.c

geniop:
	#python gsquander.py
	for FILE in us*.0.iop; do xxd -i $$FILE $$FILE.h; done

us%:
	$(CC) $(CFLAGS) $(INCLUDES) -ous$*.0.iop $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$* -shared

testus%:
	$(CC) $(CFLAGS) $(INCLUDES) -otestus$* $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$*
 
all: geniop us2 us3 us4 us5 us6 us7 us8 us9 us10 testus2 testus3 testus4 testus5 testus6 testus7 testus8 testus9 testus10

# DO NOT DELETE THIS LINE -- make depend needs it
