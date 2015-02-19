
local vprintf = terralib.externfunction("cudart:vprintf", {&int8,&int8} -> int)
local function createbuffer(args)
    local Buf = terralib.types.newstruct()
    return quote
        var buf : Buf
        escape
            for i,e in ipairs(args) do
                local typ = e:gettype()
                local field = "_"..tonumber(i)
                typ = typ == float and double or typ
                table.insert(Buf.entries,{field,typ})
                emit quote
                   buf.[field] = e
                end
            end
        end
    in
        [&int8](&buf)
    end
end

-- TODO: Make this work across platforms?
local libdevice = terralib.linklibrary("/usr/local/cuda/nvvm/libdevice/libdevice.compute_35.10.bc")

local function makemathfn(name, nargs)
	nargs = nargs or 1
	-- double version
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(double) end
	local fn = terralib.externfunction(string.format("__nv_%s", fname), ptype->double)
	-- float version
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(float) end
	fn:adddefinition(
		terralib.externfunction(string.format("__nv_%sf", fname), ptype->float):getdefinitions()[1])
	return fn
end


-- Interface to cuRAND random number generation
if os.execute("cd lib/curand; make") ~= 0 then
	error("Failed to compile curand wrapper.")
end
terralib.linklibrary("lib/curand/wrapper_terra.bc")
local struct RandomState  -- just a copy of XORWOW state
{
	d: uint32
    v: uint32[5]
    bmflag: int
    bmflag_double: int
    bmextra: float
    bmextra_double: double
}

-- thread id / block id stuff
-- Extra CUDA-specific stuff
local threadIdx = 
{
	x = cudalib.nvvm_read_ptx_sreg_tid_x,
	y = cudalib.nvvm_read_ptx_sreg_tid_y,
	z = cudalib.nvvm_read_ptx_sreg_tid_z
}
local blockDim = 
{
	x = cudalib.nvvm_read_ptx_sreg_ntid_x,
	y = cudalib.nvvm_read_ptx_sreg_ntid_y,
	z = cudalib.nvvm_read_ptx_sreg_ntid_z
}
local blockIdx = 
{
	x = cudalib.nvvm_read_ptx_sreg_ctaid_x,
	y = cudalib.nvvm_read_ptx_sreg_ctaid_y,
	z = cudalib.nvvm_read_ptx_sreg_ctaid_z
}
local gridDim = 
{
	x = cudalib.nvvm_read_ptx_sreg_nctaid_x
	y = cudalib.nvvm_read_ptx_sreg_nctaid_y
	z = cudalib.nvvm_read_ptx_sreg_nctaid_z
}
local warpSize = cudalib.nvvm_read_ptx_sreg_warpsize
local syncthreads = cudalib.nvvm_barrier0

-- Metatable for 'global' (i.e. per-semantic-thread) variables.
-- Automatically does the thread-indexing logic behind-the-scenes.
-- This creates a constant-memory pointer that can be set to refer
--    to some cudaMalloc'ed memory after a kernel has been compiled.
local globalmt = 
{
	get = function(self)
		return `[self.const] [ blockIdx.x() * blockDim.x() + threadIdx.x() ]
	end,
	getimpl = function(self) return self.const end
}
globalmt.__index = globalmt


local curt = terralib.includec("cuda_runtime.h")


return 
{
	name = "cuda",
	std = 
	{
		malloc = terralib.externfunction("malloc", {uint64} -> &opaque),
		free = terralib.externfunction("free", {&opaque} -> {}),
		memcpy = terralib.externfunction("memcpy", {&opaque, &opaque, uint64} -> {&opaque}),
		assert = macro(function(check) error("assert not (yet?) implemented for CUDA platform")),
		printf = macro(function(fmt,...)
		    local buf = createbuffer({...})
		    return `vprintf(fmt,buf) 
		end),

		-- Platforms that represent CPU-driven co-processors provide this function
		mempcyToHost = terra(dst: &opaque, src: &opaque, size: uint64)
			return curt.cudaMemcpy(dst, src, size, curt.cudaMemcpyDeviceToHost)
		end
	},

	rand = 
	{
		State = RandomState,
		init = terralib.externfunction("rand_init", {uint64, uint64, uint64, &RandomState} -> {}),
		uniform = terralib.externfunction("rand_uniform", {&RandomState} -> {double})
	},

	maths = 
	{
		acos = makemathfn("acos"),
		acosh = makemathfn("acosh"),
		asin = makemathfn("asin"),
		asinh = makemathfn("asinh"),
		atan = makemathfn("atan"),
		atan2 = makemathfn("atan2", 2),
		ceil = makemathfn("ceil"),
		cos = makemathfn("cos"),
		cosh = makemathfn("cosh"),
		exp = makemathfn("exp"),
		fabs = makemathfn("fabs"),
		floor = makemathfn("floor"),
		fmax = makemathfn("fmax", 2),
		fmin = makemathfn("fmin", 2),
		log = makemathfn("log"),
		log10 = makemathfn("log10"),
		pow = makemathfn("pow", 2),
		round = makemathfn("round"),
		sin = makemathfn("sin"),
		sinh = makemathfn("sinh"),
		sqrt = makemathfn("sqrt"),
		tan = makemathfn("tan"),
		tanh = makemathfn("tanh")
	},

	global = function(Type)
		local obj = { const = cudalib.constantmemory(&Type, 1) }
		setmetatable(obj, globalmt)
		return obj
	end

	-- CUDA specific
	runtime = curt
}



