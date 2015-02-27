
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>

void * real = NULL;
;
typedef uint64_t (*FnTyp)(uint64_t,uint64_t,uint64_t,uint64_t);

static int count = 0;

extern "C" uint64_t nvvmAddModuleToProgram(uint64_t a,uint64_t b,uint64_t c,uint64_t d) {
    if (!real) {
        void * handle = dlopen("/usr/local/cuda/nvvm/lib/libnvvm.dylib",RTLD_NOW);
        real = dlsym(handle, "nvvmAddModuleToProgram");
    }
    char buf[256];
    const char * name = getenv("LLVM_BC_NAME");
    sprintf(buf,"%s%d.bc",name ? name : "blob", count++);
    printf("writing %s\n",buf);
    FILE * e = fopen(buf,"w");
    fwrite((void*)b,1,c,e);
    fclose(e);
    return ((FnTyp)real)(a,b,c,d);
}

