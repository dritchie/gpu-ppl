return require("platform.module")(function(platform)

local util = require("lib.util")
local inherit = require("lib.inherit")
local S = require("lib.std")(platform)
local Vector = require("lib.vector")(platform)

-------------------------------------------------------------------------------

-- A random choice address, represented as a stack of identifiers
local Identifier = uint32
local Address = Vector(Identifier)

-------------------------------------------------------------------------------

-- A trace of all ERPs of one particular type used by a program.
local SingleTypeTrace = terralib.memoize(function(ERPType)

	-- A simple implementation using a flat list representation.

	local struct AddrChoicePair(S.Object)
	{
		addr: Address
		choice: ERPType
	}

	local params = ERPType.ParamTypes:map(function(t) return symbol(t) end)
	terra AddrChoicePair:__init(addr: &Address, [params])
		self.addr:copy(addr)
		self.choice:init([params])
	end

	local struct SingleTypeListTrace(S.Object)
	{
		choicerecs : Vector(AddrChoicePair)
	}

	terra SingleTypeListTrace:numChoices()
		return self.choicerecs:size()
	end

	terra SingleTypeListTrace:getChoice(i: uint)
		return &self.choicerecs(i).choice
	end

	terra SingleTypeListTrace:clear()
		self.choicerecs:clear()
	end

	-- Prepare for a trace run by marking all random choices as 'inactive'
	terra SingleTypeListTrace:prepareForRun()
		for i=0,self.choicerecs:size() do
			self.choicerecs(i).choice.active = false
		end
	end

	-- Clear out all choices that were not reached on the last run of the
	--    program.
	-- Returns the summed log prob of all removed choices.
	terra SingleTypeListTrace:clearUnreachables()
		var oldlp = 0.0
		for i=self.choicerecs:size()-1,-1,-1 do
			if not self.choicerecs(i).choice.active then
				oldlp = oldlp + self.choicerecs(i).choice.logprob
				self.choicerecs(i):destruct()
				self.choicerecs:remove(i)
				i = i + 1
			end
		end
		return oldlp
	end

	-- Returns an ERP record for the given address.
	-- Returns true if the record was retrieved from the trace, false if
	--    the record did not exist and had to be created.
	params = ERPType.ParamTypes:map(function(t) return symbol(t) end)
	terra SingleTypeListTrace:lookup(addr: &Address, [params])
		-- Search for choice with same address
		for i=0,self.choicerecs:size() do
			if self.choicerecs(i).addr == @addr then
				return &self.choicerecs(i).choice, true
			end
		end
		-- No choice record found; create a new one
		var newrec = self.choicerecs:insert()
		newrec:init(addr, [params])
		return &newrec.choice, false
	end

	return SingleTypeListTrace

	-- TODO: Hash table implementation
end)


-------------------------------------------------------------------------------


-- A trace of all ERPs of all types that could be potentially used by a program
-- This is the 'base' type of all traces, in that it doesn't depend upon the
--    return type of the program being run.
-- Later, we'll define a type constructor for static subtypes of this trace that
--    are specialized on a particular program's return type.

-- We have to defer creation of this type until all ERPs have been defined and
--    registered in the erpdefs module, since the layout of the type depends
--    upon what ERPs are available.
-- So instead of defining the type eagerly, we define it lazily by wrapping
--    it in a memoized thunk.
local ERPTypes = {}
local function registerERPType(ERPType) table.insert(ERPTypes, ERPType) end
local BaseTrace = terralib.memoize(function()

	local struct BaseTrace
	{
		logprior: double
		loglikelihood: double
		logposterior: double
		newlogprob: double
		oldlogprob: double
		address: Address
	}

	-- Retrieve a reference to the trace for the currently-executing
	--    computation (Platform specific)
	local globalTrace = platform.global(&BaseTrace)
	BaseTrace.globalTrace = globalTrace

	-- Map an ERPType to the corresponding subtrace member name
	local erptype2index = {}
	for i,erpt in ipairs(ERPTypes) do erptype2index[erpt] = i end
	local function erptype2membername(ERPType)
		local i = erptype2index[ERPType]
		if not i then
			error(string.format("Unregistered ERPType: %s",
				tostring(ERPType)))
		end
		return string.format("trace%d", i)
	end

	-- Add all subtrace members to the BaseTrace struct
	for _,erpt in ipairs(ERPTypes) do
		BaseTrace.entries:insert({
			field = erptype2membername(erpt),
			type = SingleTypeTrace(erpt)
		})
	end

	-- Invoke a Lua code-gen function for all subtraces
	local function forAllSubtraces(self, fn)
		return quote
			escape
				for _,erpt in ipairs(ERPTypes) do
					emit(fn(`self.[erptype2membername(erpt)]))
				end
			end
		end
	end
	BaseTrace.forAllSubtraces = forAllSubtraces

	terra BaseTrace:numChoices()
		var n = 0
		[forAllSubtraces(self, function(trace)
			return quote n = n + trace:numChoices() end
		end)]
		return n
	end

	terra BaseTrace:pushAddress(id: Identifier)
		self.address:insert(id)
	end

	terra BaseTrace:popAddress()
		self.address:remove()
	end

	terra BaseTrace:addPriorProb(plp: double)
		self.logprior = self.logprior + plp
		self.logposterior = self.logprior + self.loglikelihood
	end

	terra BaseTrace:addFactor(fac: double)
		self.loglikelihood = self.loglikelihood + fac
		self.logposterior = self.logprior + self.loglikelihood
	end

	terra BaseTrace:enforceConstraint(pred: bool)
		if not pred then
			self.loglikelihood = -math.huge
			self.logposterior = -math.huge
		end
	end

	function BaseTrace.lookup(ERPType)
		local params = ERPType.ParamTypes:map(function(t) return symbol(t) end)
		return terra(self: &BaseTrace, [params])
			var choicerec, found = self.[erptype2membername(ERPType)]:lookup(&self.address, [params])
			if found then
				choicerec:checkForChanges([params])
			else
				self.newlogprob = self.newlogprob + choicerec.logprob
			end
			self:addPriorProb(choicerec.logprob)
			return choicerec
		end
	end

	-- Propose a change to random choice i
	-- Returns the forward and reverse probabilities of the proposal
	terra BaseTrace:proposeChangeToChoice(i: uint)
		[forAllSubtraces(self, function(trace)
			return quote
				var n = trace:numChoices()
				if i < n then
					return trace:getChoice(i):proposal()
				end
				i = i - n
			end
		end)]
	end

	return BaseTrace

end)


-------------------------------------------------------------------------------


-- Now we define the 'concrete' trace type which is specialized on the program
--    being run.
local Trace = terralib.memoize(function(program)

	-- We assume that we can freely get the program's return type
	local succ, typ = program:peektype()
	if not succ then
		error("Program return type not specified")
	end
	local ReturnType = typ.returntype

	local struct Trace(S.Object)
	{
		returnVal: ReturnType
		hasReturnVal: bool
	}
	inherit(BaseTrace(), Trace)

	local forAllSubtraces = BaseTrace().forAllSubtraces

	terra Trace:__init(doRejectInit: bool) : {}
		self.logprior = 0.0
		self.loglikelihood = 0.0
		self.logposterior = 0.0
		self.newlogprob = 0.0
		self.oldlogprob = 0.0
		self.address:init()
		self.hasReturnVal = false

		[forAllSubtraces(self, function(trace)
			return quote trace:init() end
		end)]

		if doRejectInit then
			self.logposterior = [-math.huge]
			while self.logposterior == [-math.huge] do
				[forAllSubtraces(self, function(trace)
					return quote trace:clear() end
				end)]
				self:run()
			end
		end
	end

	terra Trace:__init() : {}
		self:__init(false)
	end

	terra Trace:run()
		-- Set the current trace to be this
		var prevTrace = [BaseTrace().globalTrace:get()]
		[BaseTrace().globalTrace:get()] = self

		-- Prepare
		self.logprior = 0.0
		self.loglikelihood = 0.0
		self.newlogprob = 0.0
		self.oldlogprob = 0.0
		[forAllSubtraces(self, function(trace)
			return quote trace:prepareForRun() end
		end)]

		-- Run
		if self.hasReturnVal then S.rundestructor(self.returnVal) end
		self.returnVal = program()
		self.hasReturnVal = true

		-- Clear out choices that are no longer reachable
		[forAllSubtraces(self, function(trace)
			return quote
				self.oldlogprob = self.oldlogprob + trace:clearUnreachables()
			end
		end)]

		-- Restore previous current trace
		[BaseTrace().globalTrace:get()] = prevTrace
	end

	return Trace

end)

-------------------------------------------------------------------------------

-- Interface that other code can use to interact with the current trace


-- Trace for currently executing program
local function globalTrace()
	return BaseTrace().globalTrace
end

-- Returns false if the program is just being run forward with no inference
local function isRecordingTrace()
	return `[globalTrace():get()] ~= nil
end

-- Sample or retrieve the value of a random choice
local function lookup(ERPType)
	-- This has to be a macro to preserve correct deferred auto-destruct
	--    behavior.
	return macro(function(...)
		local params = {...}
		return quote
			var val : ERPType.ValueType
			-- If we are tracing program execution, then attempt to look up the
			--    value in the trace
			if [isRecordingTrace()] then
				var choicerec = [BaseTrace().lookup(ERPType)]([globalTrace():get()], [params])
				var tmpval = choicerec:getValue()
				S.copy(val, tmpval)
			-- Otherwise, just sample a value directly
			else
				val = ERPType.sample([params])
			end
			-- Set up an eventual destructor call for this value, if the value type
			--    has a destructor
			escape
				if ERPType.ValueType:isstruct() and
				   ERPType.ValueType:getmethod("destruct") then
				   emit quote defer val:destruct() end
				end
			end
		in
			-- Return pointer-to-struct, in keeping with salloc() convention,
			--    if value type is not POD.
			[util.isPOD(ERPType.ValueType) and (`val) or (`&val)]
		end
	end)
end

return
{
	Trace = Trace,
	registerERPType = registerERPType,
	globalTrace = globalTrace,
	isRecordingTrace = isRecordingTrace,
	lookup = lookup
}

end)

