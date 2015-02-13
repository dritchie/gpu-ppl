
-- Super-basic stuff for now; eventually copy over more of the Quicksand
--    standard test suite.

local function test(platform)

	local p = require("prob")(platform)
	local S = require("lib.std")(platform)

	local terra model() : bool
		var x = p.flip(0.5)
		var y = p.flip(0.5)
		p.condition(x and y)
		return (x or y)
	end

	local terra go()
		var samps = [S.Vector(p.Sample(model))].salloc():init()
		[p.mh(model)](samps, 1000, 0, 1, true)
	end

	go()

end

test(require("platform.x86"))