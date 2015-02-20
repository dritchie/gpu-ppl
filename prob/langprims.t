return require("platform.module")(function(platform)

local trace = require("prob.trace")(platform)


-- Probabilistic language primitives

local id = 0

-- Wrap a function so that all calls do proper address tracking
-- TODO: Handle struct methods specially, as in Quicksand?
local function fn(funcOrMacro)
	local data = {def = nil}
	local wrappedfn = macro(function(...)
		local args = terralib.newlist({...})
		local argtypes = args:map(function(x) return x:gettype() end)
		local argstmp = argtypes:map(function(t) return symbol(t) end)
		local myid = id
		id = id + 1
		return quote
			var [argstmp] = [args]
			var hastrace = [trace.isRecordingTrace()]
			if hastrace then
				[trace.globalTrace():get()]:pushAddress(myid)
			end
			var result = [data.def]([argstmp])
			if hastrace then
				[trace.globalTrace():get()]:popAddress()
			end
		in
			result
		end
	end)
	wrappedfn.data = data
	wrappedfn.define = function(self, funcOrMacro)
		if funcOrMacro ~= nil then
			assert(terralib.isfunction(funcOrMacro) or terralib.ismacro(funcOrMacro),
				"Argument to fn:define must be a Terra function or macro")
		end
		self.data.def = funcOrMacro
	end
	wrappedfn:define(funcOrMacro)
	return wrappedfn
end


-- Modulate the probability of the computation
local factor = macro(function(num)
	return quote
		if [trace.isRecordingTrace()] then
			[trace.globalTrace():get()]:addFactor(num)
		end
	end
end)


-- Enforce a hard constraint
local condition = macro(function(pred)
	return quote
		if [trace.isRecordingTrace()] then
			[trace.globalTrace():get()]:enforceConstraint(pred)
		end
	end
end)


return 
{
	fn = fn,
	factor = factor,
	condition = condition
}

end)
