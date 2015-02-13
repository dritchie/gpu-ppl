
-- Super-basic stuff for now; eventually copy over more of the Quicksand
--    standard test suite.

local function test(platform)

local p = require("prob")(platform)
local S = require("lib.std")(platform)
local maths = require("lib.maths")()

local function expectationTest(name, prog, trueExp)

	local numsamps = 150
	local lag = 20
	local runs = 5
	local errtol = 0.07

	local terra getEstimate()
		var samps = [S.Vector(p.Sample(prog))].salloc():init()
		var est = 0.0
		var err = 0.0
		for i=0,runs do
			[p.mh(prog)](samps, numsamps, 0, lag, false)
			var mean = 0.0
			-- S.printf("------------\n")
			for j=0,numsamps do
				mean = mean + double(samps(j).value)
				-- S.printf("%g,   ", samps(j).value)
			end
			-- S.printf("\n")
			mean = mean / numsamps
			est = est + mean
			err = err + maths.fabs(est - trueExp)
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

expectationTest(
"flip expectation",
terra() : bool
	return p.flip(0.7)
end,
0.7
)

end

test(require("platform.x86"))