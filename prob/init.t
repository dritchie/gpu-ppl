return require("platform.module")(function(platform)

-- Main module that exposes all the 'public' functionality

local prob = {}

local function processModules(modnames)
	for _,modname in ipairs(modnames) do
		local mod = require(modname)(platform)
		for k,v in pairs(mod) do
			prob[k] = v
		end
	end
end

processModules({
	"prob.langprims",
	"prob.erpdefs",
	"prob.infer"
})

return prob

end)
