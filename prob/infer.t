return require("platform.module")(function(platform)

local S = require("lib.std")(platform)
local util = require("lib.util")(platform)
local rand = require("lib.rand")(platform)
local maths = require("lib.maths")(platform)
local trace = require("prob.trace")(platform)

-------------------------------------------------------------------------------

-- A sample from a probabilistic program
local Sample = S.memoize(function(program)

	-- We assume that we can freely get the program's return type
	local succ, typ = program:peektype()
	if not succ then
		error("Program return type not specified")
	end
	local ReturnType = typ.returntype

	-- Get the trace type for the program
	local TraceType = trace.Trace(program)

	local struct Sample(S.Object)
	{
		value: ReturnType,
		logprior: double,
		loglikelihood: double,
		logposterior: double
	}

	terra Sample:__init(tr: &TraceType)
		S.copy(self.value, tr.returnVal)
		self.logprior = tr.logprior
		self.loglikelihood = tr.loglikelihood
		self.logposterior = tr.logposterior
	end

	return Sample

end)

-------------------------------------------------------------------------------

-- Do lightweight MH
local mh = S.memoize(function(program)

	local TraceType = trace.Trace(program)

	-- Lightweight MH transition kernel
	-- Modifies currTrace in place
	-- Returns true if accepted proposal, false if rejected
	local terra mhKernel(currTrace: &TraceType)
		var nextTrace = TraceType.salloc():copy(currTrace)
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
			util.swap(@currTrace, @nextTrace)
			return true
		else
			return false
		end
	end

	-- MCMC main loop function
	-- Returns number of accepted proposals
	local terra mhloop(outsamps: &Sample(program),
				 numsamps: uint, burnin: uint, lag: uint, verbose: bool)
		var iters = burnin + (numsamps * lag)
		var currTrace = TraceType.salloc():init(true)
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
			if mhKernel(currTrace) then nAccepted = nAccepted + 1 end
			if i >= burnin and i % lag == 0 then
				outsamps[i / lag]:init(currTrace)
			end
		end
		return nAccepted
	end

	-- How inference actually launches is platform specific.
	if platform.name == "x86" then

		return terra(outsamps: &S.Vector(Sample(program)),
					 numsamps: uint, burnin: uint, lag: uint, seed: uint, verbose: bool)
			var iters = burnin + (numsamps * lag)
			outsamps:resize(numsamps)
			rand.init(seed, &[rand.globalState:get()])
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

		error("MH for CUDA not implemented yet")

	else
		error("Can't compile MH for unknown platform " .. platform.name)
	end

end)

-------------------------------------------------------------------------------

return 
{
	Sample = Sample,
	mh = mh
}

end)

