local maths = require("lib.maths")()
local Vector = require("lib.vector")()
local Sample = require("prob.sample")()
local infer = require("prob.infer")



local function doTests(platform)

local function expectationTest(name, progmodule, trueExp)

	local numsamps = 200
	local lag = 20
	-- local runs = 5
	local runs = 1
	local errtol = 0.07
	local randomseed = 42
	-- local verbose = false
	local verbose = true
	-- local numthreads = 32
	local numthreads = 1

	-- We assume that we can freely get the program's return type
	local hostprog = progmodule()
	local succ, typ = hostprog:peektype()
	if not succ then
		error("Program return type not specified")
	end
	local ReturnType = typ.returntype

	local terra getEstimate()
		var samps = [Vector(Sample(ReturnType))].salloc():init()
		var est = 0.0
		var err = 0.0
		for i=0,runs do
			escape
				if platform.name == "x86" then
					emit `[infer(platform).mh(progmodule)](samps, numsamps, 0, lag, randomseed, verbose)
				elseif platform.name == "cuda" then
					emit `[infer(platform).mh(progmodule)](samps, numthreads, numsamps, 0, lag, randomseed, verbose)
				end
			end
			var mean = 0.0
			for j=0,samps:size() do
				mean = mean + double(samps(j).value)
			end
			mean = mean / samps:size()
			est = est + mean
			err = err + maths.fabs(mean - trueExp)
			samps:clear()
		end
		est = est / runs
		err = err / runs
		return est, err
	end

	io.write(string.format("test: %s...", name))
	local testExp, err = unpacktuple(getEstimate())
	if err > errtol then
		print(string.format("FAILED! Expected value was %g, should have been %g", testExp, trueExp))
	else
		print("passed.")
	end

end

------------------------------------------------------------------------------

local pmodule = require("prob.module")

expectationTest(
"flip expectation",
pmodule(function(platform)
	return terra() : bool
		return p.flip(0.7)
	end
end),
0.7
)

expectationTest(
"uniform expectation",
pmodule(function(platform)
	return terra() : double
		return p.uniform(0.1, 0.4)
	end
end),
0.5*(.1+.4)
)

expectationTest(
"categorical expectation",
pmodule(function(platform)
	local Vector = require("lib.vector")(platform)
	return terra() : double
		var items = array(0.2, 0.3, 0.4)
		var params = [Vector(double)].salloc():init()
		params:insert(0.2); params:insert(0.6); params:insert(0.2)
		return items[p.categorical(params)]
	end
end),
0.2*.2 + 0.6*.3 + 0.2*.4
)

expectationTest(
"gaussian expectation",
pmodule(function(platform)
	return terra() : double
		return p.gaussian(0.1, 0.5)
	end
end),
0.1
)

expectationTest(
"gamma expectation",
pmodule(function(platform)
	return terra() : double
		return p.gamma(2.0, 2.0)/10.0
	end
end),
0.4
)

expectationTest(
"beta expectation",
pmodule(function(platform)
	return terra() : double
		return p.beta(2.0, 5.0)
	end
end),
2.0/(2.0+5.0)
)

expectationTest(
"binomial expectation",
pmodule(function(platform)
	return terra() : double
		return p.binomial(0.5, 40)/40.0
	end
end),
0.5
)

expectationTest(
"poisson expectation",
pmodule(function(platform)
	return terra() : double
		return p.poisson(4.0)/10.0
	end
end),
0.4
)

expectationTest(
"condition expectation",
pmodule(function(platform)
	return terra() : bool
		var a = p.flip(0.5)
		p.condition(a)
		return a
	end
end),
1.0
)

expectationTest(
"factor expectation",
pmodule(function(platform)
	local distrib = require("prob.distrib")(platform)
	return terra() : double
		var x = p.uniform(-1.0, 1.0)
		p.factor([distrib.gaussian(double)].logprob(x, 0.3, 0.1))
		return x
	end
end),
0.3
)

expectationTest(
"multiple choices expectation",
pmodule(function(platform)
	return terra() : bool
		var a = p.flip(0.5)
		var b = p.flip(0.5)
		p.condition(a or b)
		return (a and b)
	end
end),
1.0/3.0
)

expectationTest(
"control flow (1) expectation",
pmodule(function(platform)
	return terra() : bool
		if p.flip(0.7) then
			return p.flip(0.2)
		else
			return p.flip(0.8)
		end
	end
end),
0.7*0.2 + 0.3*0.8
)

expectationTest(
"control flow (2) expectation",
pmodule(function(platform)
	return terra() : bool
		var weight = 0.8
		if p.flip(0.7) then weight = 0.2 end
		return p.flip(weight)
	end
end),
0.7*0.2 + 0.3*0.8
)

expectationTest(
"subroutine expectation",
pmodule(function(platform)
	local helper = p.fn(terra()
		return p.gaussian(0.1, 0.5)
	end)
	return terra() : double
		return helper()
	end
end),
0.1
)

expectationTest(
"recursive subroutine expectation",
pmodule(function(platform)
	local powerlaw = p.fn()
	powerlaw:define(terra(prob: double, x: int) : int
		if p.flip(prob) then
			return x
		else
			return powerlaw(prob, x+1)
		end
	end)
	return terra() : bool
		var a = powerlaw(0.3, 1)
		return a < 5
	end
end),
0.7599
)

------------------------------------------------------------------------------

end



-- doTests(require("platform.x86"))
doTests(require("platform.cuda"))




