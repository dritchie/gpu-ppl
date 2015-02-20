return function(fn)
	local mod = require("platform.module")(fn)
	return function(platform)
		platform = platform or require("platform.x86")
		local newg = {}
		for k,v in pairs(_G) do newg[k] = v end
		local oldp = rawget(_G, "p")
		rawset(_G, "p", require("prob")(platform))
		local result = mod(platform)
		rawset(_G, "p", oldp)
		return result
	end
end