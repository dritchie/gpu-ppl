return require("platform.module")(function(platform)

local util = require("lib.util")

local S = {}

-- Add platform-specific standard library functions: printf, malloc, free, etc.
for k,v in pairs(platform.std) do S[k] = v end

-------------------------------------------------------------------------------

-- Standard object interface

S.rundestructor = macro(function(self)
    local T = self:gettype()
    local function hasdtor(T) --avoid generating code for empty array destructors
        if T:isstruct() then return T:getmethod("destruct") 
        elseif T:isarray() then return hasdtor(T.type) 
        else return false end
    end
    if T:isstruct() then
        local d = T:getmethod("destruct")
        if d then
            return `self:destruct()
        end
    elseif T:isarray() and hasdtor(T) then        
        return quote
            var pa = &self
            for i = 0,T.N do
                S.rundestructor((@pa)[i])
            end
        end
    end
    return quote end
end)

local generatedtor = macro(function(self)
    local T = self:gettype()
    local stmts = terralib.newlist()
    -- First, execute any custom destructor logic
    if T.methods.__destruct then
        stmts:insert(`self:__destruct())
    end
    -- Then, no matter what, destruct all the members of this struct
    -- The default behavior is overridable by the "__destructmembers" method
    -- This can be useful for e.g. inheritance systems, where an object's
    --    dynamic type can differ from its static type.
    if T.methods.__destructmembers then
        stmts:insert(`self:__destructmembers())
    else
        local entries = T:getentries()
        for i,e in ipairs(entries) do
            if e.field then --not a union
                stmts:insert(`S.rundestructor(self.[e.field]))
            end
        end
    end
    return stmts
end)


-- Initializer
S.init = macro(function(self)
    local T = self:gettype()
    local function hasinit(T)
        if T:isstruct() then return T:getmethod("init")
        elseif T:isarray() then return hasinit(T.type)
        else return false end
    end
    if T:isstruct() and hasinit(T) then
        return `self:init()
    elseif T:isarray() and hasinit(T) then
        return quote
            var pa = &self
            for i=0,T.N do
                S.init((@pa)[i])
            end
        end
    end
    return quote end
end)

S.initmembers = macro(function(self)
    local T = self:gettype()
    local entries = T:getentries()
    return quote
        escape
            for _,e in ipairs(entries) do
                if e.field then --not a union
                    emit `S.init(self.[e.field])
                end
            end
        end
    end
end)

local generateinit = macro(function(self, ...)
    local T = self:gettype()
    local args = {...}
    if T.methods.__init then
        return `self:__init([args])
    else
        return `S.initmembers(self)
    end
end)


-- Copy initializer
S.copy = macro(function(self, other)
    local T = self:gettype()
    local function hascopy(T)
        if T:isstruct() then return T:getmethod("copy")
        elseif T:isarray() then return hascopy(T.type)
        else return false end
    end
    if T:isstruct() and hascopy(T) then
        return `self:copy(&other)
    elseif T:isarray() and hascopy(T) then
        return quote
            var pa = &self
            for i=0,T.N do
                S.copy((@pa)[i], other[i])
            end
        end
    end
    return quote
        self = other
    end
end)

S.copymembers = macro(function(self, other)
    local T = self:gettype()
    local entries = T:getentries()
    return quote
        escape
            for _,e in ipairs(entries) do
                if e.field then --not a union
                    emit `S.copy(self.[e.field], other.[e.field])
                end
            end
        end
    end
end)

local generatecopy = macro(function(self, other)
    local T = self:gettype()
    if T.methods.__copy then
        return `self:__copy(&other)
    else
        return `S.copymembers(self, other)
    end
end)


-- Analogous to copy, but should only be called on already-initialized objects
S.clone = macro(function(self, other)
    local T = self:gettype()
    local function hasclone(T)
        if T:isstruct() then return T:getmethod("clone")
        elseif T:isarray() then return hasclone(T.type)
        else return false end
    end
    if T:isstruct() and hasclone(T) then
        return `self:clone(&other)
    elseif T:isarray() and hasclone(T) then
        return quote
            var pa = &self
            for i=0,T.N do
                S.clone((@pa)[i], other[i])
            end
        end
    end
    return quote
        self = other
    end
end)

S.clonemembers = macro(function(self, other)
    local T = self:gettype()
    local entries = T:getentries()
    return quote
        escape
            for _,e in ipairs(entries) do
                if e.field then --not a union
                    emit `S.clone(self.[e.field], other.[e.field])
                end
            end
        end
    end
end)

local generateclone = macro(function(self, other)
    local T = self:gettype()
    if T.methods.__clone then
        return `self:__clone(&other)
    else
        return `S.clonemembers(self, other)
    end
end)


-- If platform is a CPU-driven coprocessor, then we also provide methods
--    for copying an object from device to host.
-- We assume that the 'src' object itself, but not any dynamic memory
--    it refers to, has already been copied to the host (thus we are
--    free to inspect its members).
if S.memcpyToHost then

    S.copyToHost = macro(function(src, dst)
        local T = src:gettype()
        local Td = dst:gettype()
        assert(T == Td or util.areStructurallyEquivalent(T, Td))
        local function hascopy(T)
            if T:isstruct() then return T:getmethod("copyToHost")
            elseif T:isarray() then return hascopy(T.type)
            else return false end
        end
        if T:isstruct() and hascopy(T) then
            return `src:copyToHost(&dst)
        elseif T:isarray() and hascopy(T) then
            return quote
                for i=0,T.N do
                    S.copyToHost(src[i], dst[i])
                end
            end
        end
        if T == Td then
            return quote
                dst = src
            end
        else
            -- Types are structurally-equivalent but not equal, so we
            --    need a 'cast-copy' here.
            return quote
                dst = @([&Td](&src))
            end
        end
    end)

    S.copyMembersToHost = macro(function(src, dst)
        local T = src:gettype()
        local Td = dst:gettype()
        assert(T == Td or util.areStructurallyEquivalent(T, Td))
        local entries = T:getentries()
        return quote
            escape
                for _,e in ipairs(entries) do
                    if e.field then  --not a union
                        emit `S.copyToHost(src.[e.field], dst.[e.field])
                    end
                end
            end
        end
    end)

end



-- standard object metatype
-- provides T.alloc(), T.salloc(), obj:destruct(), obj:delete()
-- users should define __destruct if the object has custom destruct behavior
-- destruct will call destruct on child nodes
function S.Object(T)
    --fill in special methods/macros
    terra T:delete() : {}
        self:destruct()
        S.free(self)
    end 
    terra T.methods.alloc() : &T
        return [&T](S.malloc(sizeof(T)))
    end
    T.methods.salloc = macro(function()
        return quote
            var t : T
            defer t:destruct()
        in
            &t
        end
    end)
    terra T:destruct() : {}
        generatedtor(@self)
    end
    T.methods.init = macro(function(self, ...)
        local args = {...}
        return quote
            var s = &self
            generateinit(@s, [args])
        in
            s
        end
    end)
    terra T:initmembers() : {}
        S.initmembers(@self)
    end
    terra T:copy(other: &T) : &T
        generatecopy(@self, @other)
        return self
    end
    terra T:copymembers(other: &T)
        S.copymembers(@self, @other)
    end
    terra T:clone(other: &T)
        generateclone(@self, @other)
    end

    if S.copyToHost then
        T.methods.copyToHost = macro(function(self, dst)
            if dst:gettype():ispointer() then dst = `@dst end
            if T.methods.__copyToHost then
                return `self:__copyToHost(&dst)
            else
                return `S.copyMembersToHost(self, dst)
            end
        end)
    end
end


-------------------------------------------------------------------------------


return S

end)
