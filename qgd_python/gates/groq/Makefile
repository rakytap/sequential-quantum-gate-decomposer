GROQ_DIR ?= /opt/groq/runtime

CC = g++

# define any compile-time flags
CFLAGS = -Wall -fPIC

# define any directories containing header files other than /usr/include
INCLUDES = -I$(GROQ_DIR)/include -I./usiop

# define library paths in addition to /usr/lib
LFLAGS =

# define any libraries to link into executable:
LIBS = $(GROQ_DIR)/lib/runtimelib/lib/libgroqruntime.a -lrt -pthread

# define the C source files
SRCS = groq_us.c

geniop:
	mkdir -p build
	python gsquander.py
	for FILE in usiop/us*.0.iop; do xxd -i $$FILE $$FILE.h; done

us%:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/us$*.so $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$* -shared -O3 -fPIC -Wall -g

usall:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/libqgdDFE.so $(SRCS) $(LFLAGS) $(LIBS) -shared -O3 -fPIC -Wall

usalldual:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/libqgdDFE.so $(SRCS) $(LFLAGS) $(LIBS) -shared -O3 -fPIC -Wall -DNUM_DEVICE=2

testus%:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testus$* $(SRCS) $(LFLAGS) $(LIBS) -DNUM_QBITS=$* -DTEST -O3 -Wall -g

testus:
	$(CC) $(CFLAGS) $(INCLUDES) -obuild/testus $(SRCS) $(LFLAGS) $(LIBS) -DTEST -O3 -Wall -g
 
single: geniop usall

dual: geniop usalldual

test: geniop us2 us3 us4 us5 us6 us7 us8 us9 us10 testus2 testus3 testus4 testus5 testus6 testus7 testus8 testus9 testus10

# DO NOT DELETE THIS LINE -- make depend needs it
