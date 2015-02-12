 local C = terralib.includecstring [[
    #include <stdio.h>
    #include <stdlib.h>
    inline void flushstdout() { fflush(stdout); }
]]

return 
{
	std = 
	{
		malloc = C.malloc,
		free = C.free,
		realloc = C.realloc,
		abort = C.abort,
		printf = C.printf,
		flushstdout = C.flushstdout
	},

	rand = 
	{
		rand = C.rand,
		random = terra() return C.rand() / (C.RAND_MAX + 1.0) end,
		srand = C.srand,
	},

	maths = terralib.includec("math.h"),

	getCurrTraceFn = function(TraceType)
		local gtrace = global(&TraceType, 0)
		return function() return `gtrace end
	end
}