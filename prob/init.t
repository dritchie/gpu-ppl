
-- Main module that exposes all the 'public' functionality

local prob = {}

local function processModules(modnames)
	for _,modname in ipairs(modnames)
		local mod = require(modname)
		for k,v in pairs(mod) do
			prob[k] = v
		end
	end
end

processModules(
	"prob.langprims",
	"prob.erpdefs",
	"prob.infer"
)

return prob
