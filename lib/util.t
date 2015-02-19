local U = {}


-- Is a type "plain old data," according to Standard Object conventions?
-- Used in some places to determine when something should be passed by value or by pointer
-- (POD objects pass by value, non-POD objects pass by pointer)
local function isPOD(typ)
	-- Array types are POD if their element type is POD
	if typ:isarray() then return isPOD(type.type) end
	-- Primitive types and pointer types are fine
	if typ:isprimitive() or typ:ispointer() then return true end
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


-- Test if two types are structurally equivalent
local function areStructurallyEquivalent(T1, T2)
	-- Two primitives are equivalent if they have the same size
	if T1:isprimitive() and T2:isprimitive() then
		return (sizeof(T1) == sizeof(T2))
	-- Two pointers are always equivalent, since they have the same size
	elseif T1:ispointer() and T2:ispointer() then
		return true
	-- Two arrays are equivalent if their element types are equivalent
	--    and they have the same length
	elseif T1:isarray() and T2:isarray() then
		return (T1.N == T2.N) and
			   areStructurallyEquivalent(T1.type, T2.type)
	-- Two structs are equivalent if all their fields are
	elseif T1:isstruct() and T2:isstruct() then
		for i,e in ipairs(T1.entries) do
			if not areStructurallyEquivalent(e.type, T2.entries[i].type) then
				return false
			end
		end
		return true
	-- All other cases are false
	else
		return false
	end
end
U.areStructurallyEquivalent = areStructurallyEquivalent


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