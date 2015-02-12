return require("platform.module")(function(platform)

local S = require("lib.std")(platform)

local U = {}


-- Is a type "plain old data," according to Standard Object conventions?
-- Used in some places to determine when something should be passed by value or by pointer
-- (POD objects pass by value, non-POD objects pass by pointer)
function isPOD(typ)
	-- Non-struct types are fine
	if not typ:isstruct() then return true end
	-- User-defined ctors, dtors, or copiers are a no-no
	if typ:getmethod("__init") or typ:getmethod("__destruct") or typ:getmethod("__copy") then
		return false
	end
	-- Also can't have any members that are non-POD
	for _,e in ipairs(typ.entries) do
		if not isPOD(e.type) then return false end
	end
	return true
end
U.isPOD = isPOD


-- Equality comparison that also handles arrays
local equal
equal = macro(function(a, b)
	local A = a:gettype()
	local B = b:gettype()
	if A:isarray() or B:isarray() then
		if A ~= B then return false end
		local expr = `equal(a[0], b[0])
		for i=1,A.N-1 do expr = (`expr and equal(a[ [i] ], b[ [i] ])) end
		return expr
	end
	return `a == b
end)
U.equal = equal


U.swap = macro(function(a, b)
	return quote
		var tmp = a
		a = b
		b = tmp
	end
end)


-- Generate an S.copy statement when the second argument may be pointer-to-struct
U.ptrSafeCopy = macro(function(self, other)
	return quote
		S.copy(self, [(other:gettype() == &self:gettype()) and (`@other) or other])
	end
end)


-- Exposing the underlying C implementation of terralib.currenttimeinseconds to
--    Terra code so we don't have to re-enter the Lua interpreter to do timing.
U.currenttimeinseconds = (terralib.includecstring [[
#ifdef _WIN32
#include <io.h>
#include <time.h>
#include <Windows.h>
#undef interface
#else
#include <unistd.h>
#include <sys/time.h>
#endif
double CurrentTimeInSeconds() {
#ifdef _WIN32
    static uint64_t freq = 0;
    if(freq == 0) {
        LARGE_INTEGER i;
        QueryPerformanceFrequency(&i);
        freq = i.QuadPart;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart / (double) freq;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
#endif
}
]]).CurrentTimeInSeconds


return U

end)