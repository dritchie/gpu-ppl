return require("platform.module")(function(platform)

	local R = {}

	for k,v in pairs(platform.rand) do
		R[k] = v
	end

	-- The 'global' (i.e. per-semantic-thread) RNG
	local globalState = platform.global(R.State)
	function R.globalState() return globalState end

	R.random = terra()
		return R.uniform(&[globalState:get()])
	end

	return R

end)