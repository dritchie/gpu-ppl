return require("platform.module")(function(platform)

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
    return quote
        escape
            if T.methods.__init then
                emit `self:__init([args])
            else
                emit `S.initmembers(self)
            end
        end
    end
end)



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
    local To = other:gettype()
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
    return quote
        escape
            if T.methods.__copy then
                emit `self:__copy(&other)
            else
                emit `S.copymembers(self, other)
            end
        end
    end
end)



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
end


-------------------------------------------------------------------------------


return S

end)
