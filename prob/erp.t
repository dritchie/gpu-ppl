local S = require("lib.std")
local util = require("lib.util")
local trace = require("prob.trace")
local langprims = require("prob.langprims")


-- An ERP is built from a sample function, a logprob function, and
--    (optionally) a propose function
-- ERPType is a struct type that serves as a record of an instance
--    of that ERP in a program trace.
-- The return value of this function is a macro which performs sampling /
--    trace lookup as necessary.
local function makeERP(sample, logprob, propose)

	-- Determine value and parameter types for this ERP
	local ValueType = sample:gettype().returntype
	local ParamTypes = sample:gettype().parameters
	-- Parameters that are pointer-to-struct are stored as struct value types
	local StoredParamTypes = ParamTypes:map(function(pt)
		if pt:ispointertostruct() then
			return pt.type
		else
			return pt
		end
	end)
	-- We also verify that any non-POD parameter types have a copy method
	--    (otherwise memory bugs will arise)
	for _,spt in ipairs(StoredParamTypes) do
		if not util.isPOD(spt) then
			assert(spt:getmethod("copy"),
				"Non-POD ERP params must have a 'copy' method")
		end
	end

	-- The default proposal just resamples from the prior
	if not propose then
		local ArgValueType = logprob:gettype().parameters[1]
		local params = ParamTypes:map(function(t) return symbol(t) end)
		propose = terra(currval: ArgValueType, [params])
			var newval = sample([params])
			var fwdlp = logprob([(&ValueType == ArgValueType) and
								 (`&newval) or (`newval)])
			var rvslp = logprob(currval, [params])
			return newval, fwdlp, rvslp
		end
	end

	-- Determine type layout
	local struct ERPType(S.Object)
	{
		value: ValueType,
		logprob: double,
		active: bool
	}
	local function paramField(i) = return string.format("param%d", i) end
	for i,spt in ipairs(StoredParamTypes) do
		ERPType.entries:insert({field=paramField(i), type=spt})
	end
	ERPType.ValueType = ValueType
	ERPType.ParamTypes = ParamTypes
	ERPType.methods.sample = sample
	ERPType.methods.logprob = logprob
	ERPType.method.propose = propose

	ERPType.methods.getValue = macro(function(self)
		return `self.value
	end)

	local params = ParamTypes:map(function(t) return symbol(t) end)
	terra ERPType:__init([params])
		self.active = true
		self.value = sample([params])
		escape
			for i,_ in ipairs(ParamTypes) do
				emit quote util.ptrSafeCopy(self.[paramField(i)], [params[i]]) end
			end
		end
		self:rescore()
	end

	-- When re-using a random choice record, check if any of its parameters have
	--    changed. If so, copy those changes and rescore the ERP.
	params = ParamTypes:map(function(t) return symbol(t) end)
	terra ERPType:checkForChanges([params])
		var needsRescore = false
		escape
			for i,_ in ipairs(ParamTypes) do
				local p = `[params[i]]
				-- __eq operator takes value types
				if ParamTypes[i]:ispointertostruct() then p = `@p end
				emit quote
					if not util.equal(self.[paramField(i)], p) then
						needsRescore = true
						S.rundestructor(self.[paramField(i)])
						ptrSafeCopy(self.[paramField(i)], [paramSyms[i]])
					end
				end
			end
		end
		if needsRescore then rescore() end
		self.active = true
	end

	-- Handles passing stored value + parameters to logprob / proposal functions,
	--    converting values to pointers as needed.
	local function arglist(self)
		local lst = terralib.newlist()
		if logprob:gettype().parameters[1] == &ValueType then
			lst:insert(`&self.value)
		else
			lst:insert(`self.value)
		end
		for i=1,#ParamTypes do
			if ParamTypes[i] == &StoredParamTypes[i] then
				lst:insert(`&self.[paramField(i)])
			else
				lst:insert(`self.[paramField(i)])
			end
		end
		return lst
	end

	terra ERPType:rescore()
		self.logprob = logprob([arglist(self)])
	end

	-- Modify value in place, return forward/reverse proposal probabilities
	terra ERPType:proposal()
		var newval, fwdlp, rvslp = propose([arglist(self)])
		S.rundestructor(self.value)
		self.value = newval
		return fwdlp, rvslp
	end


	-- Register ERPType with trace module
	trace.registerERPType(ERPType)


	-- Finally, this is the function that actually performs the ERP sampling
	return langprims.fn(macro(function(...)
		local params = {...}
		return `[trace.lookup(ERPType)]([params])
	end))

end


return 
{
	makeERP = makeERP
}



