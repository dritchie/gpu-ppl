-- Really simple static single inheritance
-- No dynamic dispatch. It simply does:
--    * Inheriting all members
--    * Inheriting all static methods
--    * Allows casting of subtype to supertype


-- metadata for class system
local metadata = {}

local function issubclass(child,parent)
	if child == parent then
		return true
	else
		local par = metadata[child].parent
		return par and issubclass(par,parent)
	end
end

local function setParent(child, parent)
	local md = metadata[child]
	if md then
		if md.parent then
			error(string.format("'%s' already inherits from some type -- multiple inheritance not allowed.", child.name))
		end
		md.parent = parent
	else
		metadata[child] = {parent = parent}
	end
end

local function castoperator(from, to, exp)
	if from:ispointer() and to:ispointer() and issubclass(from.type, to.type) then
		return `[to](exp)
	else
		error(string.format("'%s' does not inherit from '%s'", from.type.name, to.type.name))
	end
end

local function lookupParentStaticMethod(class, methodname)
	local cls = class
	while cls ~= nil do
		if cls.methods[methodname] ~= nil then
			return cls.methods[methodname]
		else
			if metadata[cls] and metadata[cls].parent then
				cls = metadata[cls].parent
			else
				cls = nil
			end
		end
	end
	return nil
end

local function copyparentlayoutStatic(class)
	local parent = metadata[class].parent
	for i,e in ipairs(parent.entries) do table.insert(class.entries, i, e) end
	return class.entries
end

local function addstaticmetamethods(class)
	class.metamethods.__cast = castoperator
	class.metamethods.__getentries = copyparentlayoutStatic
	class.metamethods.__getmethod = lookupParentStaticMethod
end


-- child inherits data layout and method table from parent
function staticExtend(parent, child)
	setParent(child, parent)
	addstaticmetamethods(child)
end

return staticExtend