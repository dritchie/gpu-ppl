# Should specify:
#    * LLVM_TERRA: Root directory of llvm install that Terra was built against
-include ../Makefile.inc

CLANG = $(LLVM_TERRA)/bin/clang++
AS = $(LLVM_TERRA)/bin/llvm-as

all: wrapper_terra.bc

wrapper_terra.bc: wrapper.ll
	$(AS) < $< > $@

wrapper.ll: wrapper.cpp
	$(CLANG) -std=c++11 -S -emit-llvm $< -o $@

clean:
	rm -f *.ll *.bc
