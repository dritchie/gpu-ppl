local util = require("lib.util")


return require("platform.module")(function(platform)


local rand = require("lib.rand")(platform)
local maths = require("lib.maths")(platform)
local trace = require("prob.trace")(platform)
local Sample = require("prob.sample")(platform)
local S = require("lib.std")(platform)
local Vector = require("lib.vector")(platform)


-- Do lightweight MH
local mh = terralib.memoize(function(progmodule)

	local program = progmodule(platform)

	local BaseTrace = trace.BaseTrace()
	local TraceType = trace.Trace(program)

	-- We assume that we can freely get the program's return type
	local succ, typ = program:peektype()
	if not succ then
		error("Program return type not specified")
	end
	local ReturnType = typ.returntype

	-- Lightweight MH transition kernel
	-- Modifies currTrace in place
	-- Returns true if accepted proposal, false if rejected
	local terra mhKernel(_currTrace: &&TraceType, _nextTrace: &&TraceType)
		var currTrace = @_currTrace
		var nextTrace = @_nextTrace
		nextTrace:clone(currTrace)
		var nold = nextTrace:numChoices()
		var whichi = uint(nold * rand.random())
		var fwdPropLP, rvsPropLP = nextTrace:proposeChangeToChoice(whichi)
		nextTrace:run()
		var nnew = nextTrace:numChoices()
		fwdPropLP = fwdPropLP + nextTrace.newlogprob - maths.log(double(nold))
		rvsPropLP = rvsPropLP + nextTrace.oldlogprob - maths.log(double(nnew))
		var acceptThresh = (nextTrace.logposterior - currTrace.logposterior) +
							rvsPropLP - fwdPropLP
		if maths.log(rand.random()) < acceptThresh then
			util.swap(@_currTrace, @_nextTrace)
			return true
		else
			return false
		end
	end

	-- MCMC main loop function
	-- Returns number of accepted proposals
	local terra mhloop(outsamps: &Sample(ReturnType),
				 numsamps: uint, burnin: uint, lag: uint, verbose: bool)
		var iters = burnin + (numsamps * lag)
		var currTrace = TraceType.salloc():init(true)
		var nextTrace = TraceType.salloc():init(false)
		var nAccepted = 0
		for i=0,iters do
			-- Spit out progress output if the platform supports arbitrary
			--    stdout flushing.
			escape
				if S.flushstdout then emit quote
					if verbose then
						S.printf(" Iteration %u/%u                     \r", i+1, iters)
						S.flushstdout()
					end
				end end
			end
			if mhKernel(&currTrace, &nextTrace) then nAccepted = nAccepted + 1 end
			if i >= burnin and i % lag == 0 then
				outsamps[i / lag]:init(&currTrace.returnVal, currTrace.logposterior)
			end
		end
		return nAccepted
	end

	-- How inference actually launches is platform specific.
	if platform.name == "x86" then

		return terra(outsamps: &Vector(Sample(ReturnType)),
					 numsamps: uint, burnin: uint, lag: uint, seed: uint, verbose: bool)
			var iters = burnin + (numsamps * lag)
			outsamps:resize(numsamps)
			rand.init(seed, &[rand.globalState():get()])
			var t0 = util.currenttimeinseconds()
			var nAccepted = mhloop(&outsamps(0), numsamps, burnin, lag, verbose)
			var t1 = util.currenttimeinseconds()
			if verbose then
				S.printf("\n")
				S.printf("Acceptance Ratio: %u/%u (%g%%)\n", nAccepted, iters,
					100.0*double(nAccepted)/iters)
				S.printf("Time: %g\n", t1 - t0)
			end
		end

	elseif platform.name == "cuda" then

		local cuda = platform
		local host = require("platform.x86")

		local HostS = require("lib.std")(host)
		local HostVector = require("lib.vector")(host)
		local HostSample = require("prob.sample")(host)
		local hostprogram = progmodule(host)
		succ, typ = hostprogram:peektype()
		local HostReturnType = typ.returntype

		-- The actual CUDA kernel that does the work.
		-- Each thread generates numsamps samples out output.
		local terra kernel(outsamps: &Sample(ReturnType), outnaccept: &uint,
						   numsamps: uint, burnin: uint, lag: uint, seed: uint)
			var idx = S.blockIdx.x() * S.blockDim.x() + S.threadIdx.x()
			rand.init(seed, idx, 0, &[rand.globalState():get()])
			var sampsbaseptr = outsamps + (idx * numsamps)
			var nAccepted = mhloop(sampsbaseptr, numsamps, burnin, lag, false)
			outnaccept[idx] = nAccepted
		end
   
		-- Compile it, passing in constant memory refs for 'globals'
		io.write("[[ Compiling CUDA MH kernel...")
		io.flush()
		local t0 = terralib.currenttimeinseconds()
		local CUDAmodule = terralib.cudacompile({
			kernel = kernel,
			gTraces = trace.globalTrace():getimpl(),
			gRNGS = rand.globalState():getimpl()
		})
		local t1 = terralib.currenttimeinseconds()
		print(" Done (Compile time: " .. tostring(t1-t0) .. ") ]]")

		-- Error-checking macro for CUDA invocations
		local CUDA_ERRCHECK = macro(function(stmt)
			return quote
				var retcode = stmt
				if retcode ~= cuda.runtime.cudaSuccess then
					HostS.printf("CUDA call failed with error: %s\n",
						cuda.runtime.cudaGetErrorString(retcode))
					HostS.exit(1)
				end
			end
		end)

		-- CPU-side Terra wrapper that launches kernel and packages up the results
		return terra(outsamps: &HostVector(HostSample(HostReturnType)), numblocks: uint, numthreads: uint,
					 numsamps: uint, burnin: uint, lag: uint, seed: uint, verbose: bool)

			-- Make the stack bigger so we don't overflow it.
			-- 8k seems to be enough (thus far). Note that the XORWOW rng requires more space.
			var stacksize = 4 * 1024
			cuda.runtime.cudaDeviceSetLimit(cuda.runtime.cudaLimitStackSize, stacksize)

			-- Allocate space for 'globals', point constant memory refs at this space
			var gtraces : &&BaseTrace
			var grngs : &rand.State
			cuda.runtime.cudaMalloc([&&opaque](&gtraces), sizeof([&BaseTrace])*numblocks*numthreads)
			cuda.runtime.cudaMalloc([&&opaque](&grngs), sizeof([rand.State])*numblocks*numthreads)
			cuda.runtime.cudaMemcpy(CUDAmodule.gTraces, &gtraces, sizeof([&&BaseTrace]),
									cuda.runtime.cudaMemcpyHostToDevice)
			cuda.runtime.cudaMemcpy(CUDAmodule.gRNGS, &grngs, sizeof([&rand.State]),
									cuda.runtime.cudaMemcpyHostToDevice)

			-- Allocate space for results (samples and naccept)
			var samps : &Sample(ReturnType)
			var naccepts : &uint
			cuda.runtime.cudaMalloc([&&opaque](&samps), sizeof([Sample(ReturnType)])*numsamps*numblocks*numthreads)
			cuda.runtime.cudaMalloc([&&opaque](&naccepts), sizeof(uint)*numblocks*numthreads)

			-- Launch kernel
			if verbose then
				HostS.printf("[[ Launching CUDA MH kernel ]]\n")
			end
			var launchparams = terralib.CUDAParams { numblocks,1,1, numthreads,1,1, 0, nil }
			var t0 = util.currenttimeinseconds()
			CUDA_ERRCHECK(CUDAmodule.kernel(&launchparams, samps, naccepts, numsamps, burnin, lag, seed))
			CUDA_ERRCHECK(cuda.runtime.cudaDeviceSynchronize())
			var t1 = util.currenttimeinseconds()

			if verbose then
				-- Copy naccepts to host
				var hostnaccepts = [&uint](HostS.malloc(sizeof(uint)*numblocks*numthreads))
				S.memcpyToHost(hostnaccepts, naccepts, sizeof(uint)*numblocks*numthreads)
				var nTotal = (burnin + (numsamps * lag)) * numblocks * numthreads
				var nAccepted = 0
				for i=0,numblocks*numthreads do
					nAccepted = nAccepted + hostnaccepts[i]
				end
				-- Report stuff
				HostS.printf("Acceptance Ratio: %u/%u (%g%%)\n", nAccepted, nTotal,
					100.0*double(nAccepted)/nTotal)
				HostS.printf("Time: %g\n", t1 - t0)
				HostS.free(hostnaccepts)
			end

			-- Copy samples to host
			var tmpsamps = [&Sample(ReturnType)](HostS.malloc(sizeof([Sample(ReturnType)])*numsamps*numblocks*numthreads))
			S.memcpyToHost(tmpsamps, samps, sizeof([Sample(ReturnType)])*numsamps*numblocks*numthreads)
			outsamps:resize(numsamps*numblocks*numthreads)
			for i=0,numsamps*numblocks*numthreads do
				tmpsamps[i]:copyToHost(outsamps:get(i))
			end
			HostS.free(tmpsamps)

			-- Cleanup
			cuda.runtime.cudaFree(gtraces)
			cuda.runtime.cudaFree(grngs)
			cuda.runtime.cudaFree(samps)
			cuda.runtime.cudaFree(naccepts)
		end

	else
		error("Can't compile MH for unknown platform " .. platform.name)
	end

end)

-------------------------------------------------------------------------------

return 
{
	mh = mh
}

end)

