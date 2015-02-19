return require("platform.module")(function(platform)

local S = require("lib.std")(platform)

-- A sample from a probabilistic program
return terralib.memoize(function(ValueType)

	local struct Sample(S.Object)
	{
		value: ValueType,
		logprob: double
	}

	terra Sample:__init(value: &ValueType, logprob: double)
		S.copy(self.value, @value)
		self.logprob = logprob
	end

	return Sample

end)

end)