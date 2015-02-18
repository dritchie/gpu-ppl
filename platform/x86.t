 local C = terralib.includecstring [[
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    inline void flushstdout() { fflush(stdout); }
]]

-- Interface to external C++ random number generator
if os.execute("cd lib/stdrand; make") ~= 0 then
	error("Failed to compile stdrand wrapper.")
end
terralib.linklibrary("lib/stdrand/wrapper_terra.bc")
local struct RandomState
{
	pad: uint8[ terralib.externfunction("rand_state_size", {}->{uint64})() ]
}

-- Global variables metatable
local globalmt = 
{
	get = function(self) return self.global end,
	getimpl = function(self) return self.global end
}
globalmt.__index = globalmt

return 
{
	std = 
	{
		malloc = C.malloc,
		free = C.free,
		memcpy = C.memcpy,
		realloc = C.realloc,
		assert = macro(function(check)
		    local loc = check.tree.filename..":"..check.tree.linenumber
		    return quote 
		        if not check then
		            C.printf("%s: assertion failed!\n",loc)
		            C.abort()
		        end
		    end
		end),
		printf = C.printf,
		flushstdout = C.flushstdout
	},

	rand = 
	{
		State = RandomState,
		init = terralib.externfunction("rand_init", {uint32, &RandomState} -> {}),
		uniform = terralib.externfunction("rand_uniform", {&RandomState} -> {double})
	},

	maths = terralib.includec("math.h"),

	global = function(Type)
		local obj = { global = global(Type) }
		setmetatable(obj, globalmt)
		return obj
	end,
}