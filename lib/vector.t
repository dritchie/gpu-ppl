return require("platform.module")(function(platform)

local S = require("lib.std")(platform)

return terralib.memoize(function(T,debug)
    local struct Vector(S.Object) {
        _data : &T;
        _size : uint64;
        _capacity : uint64;
    }
    function Vector.metamethods.__typename() return ("Vector(%s)"):format(tostring(T)) end
    Vector.isstdvector = true
    Vector.type = T
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Vector:__init() : {}
        self._data,self._size,self._capacity = nil,0,0
    end
    terra Vector:__init(cap : uint64) : {}
        self:__init()
        self:reserve(cap)
    end
    terra Vector:reserve(cap : uint64)
        if cap > 0 and cap > self._capacity then
            var oc = self._capacity
            if self._capacity == 0 then
                self._capacity = 16
            end
            while self._capacity < cap do
                self._capacity = self._capacity * 2
            end
            escape
                -- Use realloc if platform provides it
                if S.realloc then
                    emit quote
                        self._data = [&T](S.realloc(self._data,sizeof(T)*self._capacity))
                    end
                else
                    emit quote
                        var newdata = [&T](S.malloc(sizeof(T)*self._capacity))
                        if self._data ~= nil then
                            S.memcpy(newdata, self._data, sizeof(T)*self._size)
                            -- for i=0,self._size do newdata[i] = self._data[i] end
                            S.free(self._data)
                        end
                        self._data = newdata
                    end
                end
            end
        end
    end
    terra Vector:resize(size: uint64)
        self:reserve(size)
        self._size = size
    end
    terra Vector:__destruct()
        self:clear()
        if self._data ~= nil then
            S.free(self._data)
            self._data = nil
            self._capacity = 0
        end
    end
    terra Vector:size() return self._size end
    
    terra Vector:get(i : uint64)
        assert(i < self._size) 
        return &self._data[i]
    end
    Vector.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Vector:insert(idx : uint64, N : uint64, v : T) : {}
        assert(idx <= self._size)
        self._size = self._size + N
        self:reserve(self._size)
        
        if self._size > N then
            var i = self._size
            while i > idx do
                self._data[i - 1] = self._data[i - 1 - N]
                i = i - 1
            end
        end
        
        for i = 0ULL,N do
            self._data[idx + i] = v
        end
    end
    terra Vector:insert(idx : uint64, v : T) : {}
        return self:insert(idx,1,v)
    end
    terra Vector:insert(v : T) : {}
        return self:insert(self._size,1,v)
    end
    terra Vector:insert() : &T
        self._size = self._size + 1
        self:reserve(self._size)
        return self:get(self._size - 1)
    end
    terra Vector:remove(idx : uint64) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end
    terra Vector:remove() : T
        assert(self._size > 0)
        return self:remove(self._size - 1)
    end

    terra Vector:clear() : {}
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        self._size = 0
    end

    terra Vector:__copy(other: &Vector) : {}
        self:__init(other:size())
        for i=0,other:size() do
            self:insert()
            S.copy(self(i), other(i))
        end
    end

    -- Device-to-host copy for coprocessor platforms
    if S.copyToHost then
        local hostplatform = require("platform.x86")
        local HostVector = require("lib.vector")(hostplatform)(T)
        terra Vector:__copyToHost(dst: &HostVector)
            dst:resize(self:size())
            var tmpmem = [&T](S.malloc(sizeof(T)*self:size()))
            S.memcpyToHost(tmpmem, self._data, sizeof(T)*self:size())
            for i=0,self:size() do
                S.copyToHost(tmpmem[i], dst(i))
            end
            S.free(tmpmem)
        end
    end

    Vector.metamethods.__eq = terra(self: Vector, other: Vector) : bool
        if self:size() ~= other:size() then return false end
        for i=0,self:size() do
            if not (self(i) == other(i)) then return false end
        end
        return true
    end
    Vector.metamethods.__eq:setinlined(true)

    Vector.metamethods.__for = function(syms, iter, body)
        local e = symbol()
        return {`@e}, quote
            var self = iter
            for i=0,self:size() do
                var [e] = self:get(i)
                body
            end
        end
    end
    
    return Vector
end)

end)


