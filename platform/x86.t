 local C = terralib.includecstring [[
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    inline void flushstdout() { fflush(stdout); }
]]

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
		flushstdout = C.flushstdout 	-- TODO: get rid of this somehow...
	},

	rand = 
	{
		random = terra() return C.rand() / (C.RAND_MAX + 1.0) end,
		srand = C.srand,
	},

	maths = terralib.includec("math.h"),

	getCurrTraceFn = function(TraceType)
		local gtrace = global(&TraceType, 0)
		return function() return `gtrace end
	end
}