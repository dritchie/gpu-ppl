local x86 = require("platform.x86")
local cuda = require("platform.cuda")
local rand = require("lib.rand")(cuda)
local S = require("lib.std")(x86)


local seed = 42


local terra kernel(outdata: &double)
	var rstate : rand.State
	var idx = cuda.std.blockIdx.x() * cuda.std.blockDim.x() + cuda.std.threadIdx.x()
	rand.init(seed, idx, 0, &rstate)
	outdata[idx] = rand.uniform(&rstate)
end

local Cmod = terralib.cudacompile({kernel = kernel}, true)

local CUDA_ERRCHECK = macro(function(stmt)
	return quote
		var retcode = stmt
		if retcode ~= cuda.runtime.cudaSuccess then
			S.printf("CUDA call failed with error: %s\n",
				cuda.runtime.cudaGetErrorString(retcode))
			S.exit(1)
		end
	end
end)

local N = 1
local terra test()
	var outdata : &double
	cuda.runtime.cudaMalloc([&&opaque](&outdata), sizeof(double)*N)
	var launchparams = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
	CUDA_ERRCHECK(Cmod.kernel(&launchparams, outdata))
	CUDA_ERRCHECK(cuda.runtime.cudaDeviceSynchronize())
	cuda.runtime.cudaFree(outdata)
end

-- test()