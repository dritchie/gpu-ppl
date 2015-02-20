
-- Macro to give usual interface to printf
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


------------------------------------------------------------------------------


-- Math functions
-- TODO: Make this work across platforms?
local libdevice = terralib.linklibrary("/usr/local/cuda/nvvm/libdevice/libdevice.compute_35.10.bc")
local mathfns = {}
local function addmathfn(name, nargs)
	nargs = nargs or 1
	-- double version
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(double) end
	mathfns[name] = terralib.externfunction(string.format("__nv_%s", name), ptype->double)
	-- float version
	local ptype = terralib.newlist()
	for i=1,nargs do ptype:insert(float) end
	mathfns[name.."f"] = terralib.externfunction(string.format("__nv_%sf", name), ptype->float)
end
addmathfn("acos")
addmathfn("acosh")
addmathfn("asin")
addmathfn("asinh")
addmathfn("atan")
addmathfn("atan2", 2)
addmathfn("ceil")
addmathfn("cos")
addmathfn("cosh")
addmathfn("exp")
addmathfn("fabs")
addmathfn("floor")
addmathfn("fmax", 2)
addmathfn("fmin", 2)
addmathfn("log")
addmathfn("log10")
addmathfn("pow", 2)
addmathfn("round")
addmathfn("sin")
addmathfn("sinh")
addmathfn("sqrt")
addmathfn("tan")
addmathfn("tanh")


------------------------------------------------------------------------------


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


------------------------------------------------------------------------------


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
	x = cudalib.nvvm_read_ptx_sreg_nctaid_x,
	y = cudalib.nvvm_read_ptx_sreg_nctaid_y,
	z = cudalib.nvvm_read_ptx_sreg_nctaid_z
}
local warpSize = cudalib.nvvm_read_ptx_sreg_warpsize
local syncthreads = cudalib.nvvm_barrier0


------------------------------------------------------------------------------


-- Metatable for 'global' (i.e. per-semantic-thread) variables.
-- Automatically does the thread-indexing logic behind-the-scenes.
-- This creates a constant-memory pointer that can be set to refer
--    to some cudaMalloc'ed memory after a kernel has been compiled.
local globalmt = 
{
	get = function(self)
		local c = self.const
		-- c[0] because c is an array of constant memory with one element
		return `c[0][ blockIdx.x() * blockDim.x() + threadIdx.x() ]
	end,
	getimpl = function(self) return self.const end
}
globalmt.__index = globalmt


------------------------------------------------------------------------------


-- CUDA runtime library
local curt = terralib.includec("cuda_runtime.h")


------------------------------------------------------------------------------


return 
{
	name = "cuda",
	std = 
	{
		malloc = terralib.externfunction("malloc", {uint64} -> &opaque),
		free = terralib.externfunction("free", {&opaque} -> {}),
		memcpy = terralib.externfunction("memcpy", {&opaque, &opaque, uint64} -> {&opaque}),
		assert = macro(function(check) error("assert not (yet?) implemented for CUDA platform") end),
		printf = macro(function(fmt,...)
		    local buf = createbuffer({...})
		    return `vprintf(fmt,buf) 
		end),

		threadIdx = threadIdx,
		blockDim = blockDim,
		blockIdx = blockIdx,
		gridDim = gridDim,
		warpSize = warpSize,
		syncthreads = syncthreads,

		-- Platforms that represent CPU-driven co-processors provide this function
		mempcyToHost = terra(dst: &opaque, src: &opaque, size: uint64)
			return curt.cudaMemcpy(dst, src, size, curt.cudaMemcpyDeviceToHost)
		end
	},

	rand = 
	{
		State = RandomState,
		init = terralib.externfunction("cu_rand_init", {uint64, uint64, uint64, &RandomState} -> {}),
		uniform = terralib.externfunction("cu_rand_uniform", {&RandomState} -> {double})
	},

	maths = mathfns,

	global = function(Type)
		local obj = { const = cudalib.constantmemory(&Type, 1) }
		setmetatable(obj, globalmt)
		return obj
	end,

	-- CUDA specific
	runtime = curt
}



