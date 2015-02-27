# Should specify:
#    * LLVM_31: Root directory of llvm_31 install
#    * LLVM_TERRA: Root directory of llvm install that Terra was built against
# Optionally specifies:
#    * CUDA_ROOT: Root directory of cuda install (defaults to /usr/local/cuda)
#    * CUDA_ARCH: Target architecture for building cuda code (defaults to compute_30)
-include ../Makefile.inc

UNAME = $(shell uname)

CLANG = $(LLVM_TERRA)/bin/clang++
DIS = $(LLVM_31)/bin/llvm-dis
AS = $(LLVM_TERRA)/bin/llvm-as
CUDA_ROOT ?= /usr/local/cuda
CUDA_ARCH ?= compute_30

ifeq ($(UNAME), Darwin)
LIBDUMP = libdump.dylib
NVCC_PRELOAD = DYLD_INSERT_LIBRARIES=$(shell pwd)/$(LIBDUMP) DYLD_FORCE_FLAT_NAMESPACE=1
else
LIBDUMP = libdump.so
NVCC_PRELOAD = LD_PRELOAD=$(shell pwd)/$(LIBDUMP)
endif

all: wrapper_terra.bc

clean:
	rm -f *.dylib *.so *.ptx *.ll *.bc

$(LIBDUMP): dump.cpp
	$(CLANG) -fPIC -shared $< -o $@

wrapper0.bc: wrapper.cu $(LIBDUMP)
	$(NVCC_PRELOAD) LLVM_BC_NAME="wrapper" $(CUDA_ROOT)/bin/nvcc -ptx wrapper.cu -arch=$(CUDA_ARCH) -o wrapper.ptx

wrapper.ll: wrapper0.bc
	$(DIS) < $< > $@

wrapper_cleaned.ll: wrapper.ll
	$(shell grep -v "datalayout" $< | grep -v "llvm.used" > $@)

wrapper_terra.bc: wrapper_cleaned.ll
	$(AS) < $< > $@
