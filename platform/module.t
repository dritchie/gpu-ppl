-- Memoize a function that takes a platform and returns a module of Terra code.
-- If no platform provided, defaults to x86.
return function(fn)
	local memfn = terralib.memoize(fn)
	return function(platform)
		platform = platform or require("platform.x86")
		return memfn(platform)
	end
end