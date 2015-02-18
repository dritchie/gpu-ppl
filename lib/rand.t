return require("platform.module")(function(platform)

	local R = {}

	for k,v in pairs(platform.rand) do
		R[k] = v
	end

	-- The 'global' (i.e. per-semantic-thread) RNG
	R.globalState = platform.global(R.State)

	R.random = terra()
		return R.uniform(&[R.globalState:get()])
	end

	return R

end)