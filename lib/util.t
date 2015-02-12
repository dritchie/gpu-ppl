local S = require("lib.std")

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


return U