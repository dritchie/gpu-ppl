return require("platform.module")(function(platform)

local S = require("lib.std")(platform)
local maths = require("lib.maths")(platform)
local R = require("lib.rand")(platform)


local D = {}

--------------------------------------------

D.bernoulli = S.memoize(function(real)
	return {
		sample = terra(p: real) : bool
			var randval = R.random()
			return (randval < p)
		end,
		logprob = terra(val: bool, p: real) : real
			var prob: real
			if val then
				prob = p
			else
				prob = 1.0 - p
			end
			return maths.log(prob)
		end
	}
end)

--------------------------------------------

D.uniform = S.memoize(function(real)
	return {
		sample = terra(lo: real, hi: real) : real
			var u = R.random()
			return (1.0-u)*lo + u*hi
		end,
		logprob = terra(val: real, lo: real, hi: real) : real
			if val < lo or val > hi then return [-math.huge] end
			return -maths.log(hi - lo)
		end
	}
end)

--------------------------------------------

D.uniformInt = S.memoize(function(real)
	return {
		sample = terra(lo: int, hi: int) : int
			var u = R.random()
			return int((1.0-u)*lo + u*hi)
		end,
		logprob = terra(val: int, lo: int, hi: int) : real
			if val < lo or val >= hi then return [-math.huge] end
			return -maths.log(hi - lo)
		end
	}
end)

--------------------------------------------

D.gaussian = S.memoize(function(real)
	local flt = double
	return {
		sample = terra(mu: real, sigma: real) : real
			var u:flt, v:flt, x:flt, y:flt, q:flt
			repeat
				u = 1.0 - R.random()
				v = 1.7156 * (R.random() - 0.5)
				x = u - 0.449871
				y = maths.fabs(v) + 0.386595
				q = x*x + y*(0.196*y - 0.25472*x)
			until not(q >= 0.27597 and (q > 0.27846 or v*v > -4 * u * u * maths.log(u)))
			return mu + sigma*v/u
		end,
		logprob = terra(x: real, mu: real, sigma: real) : real
			var xminusmu = x - mu
			return -.5*(1.8378770664093453 + 2*maths.log(sigma) + xminusmu*xminusmu/(sigma*sigma))
		end
	}
end)

--------------------------------------------

local gamma_cof = global(double[6])
local terra init_gamma_cof()
	gamma_cof = array(76.18009172947146,
					  -86.50532032941677,
					  24.01409824083091,
					  -1.231739572450155,
					  0.1208650973866179e-2,
					  -0.5395239384953e-5)
end
init_gamma_cof()
local log_gamma = S.memoize(function(real)
	return terra(xx: real)
		var x = xx - 1.0
		var tmp = x + 5.5
		tmp = tmp - (x + 0.5)*maths.log(tmp)
		var ser = real(1.000000000190015)
		for j=0,5 do
			x = x + 1.0
			ser = ser + gamma_cof[j] / x
		end
		return -tmp + maths.log(2.5066282746310005*ser)
	end
end)


D.gamma = S.memoize(function(real)
	local flt = double
	local terra sample(shape: real, scale: real) : real
		if shape < 1.0 then return sample(1.0+shape,scale) * maths.pow(R.random(), 1.0/shape) end
		var x:flt, v:real, u:flt
		var d = shape - 1.0/3.0
		var c = 1.0/maths.sqrt(9.0*d)
		while true do
			repeat
				x = [D.gaussian(flt)].sample(0.0, 1.0)
				v = 1.0+c*x
			until v > 0.0
			v = v*v*v
			u = R.random()
			if (u < 1.0 - .331*x*x*x*x) or (maths.log(u) < .5*x*x + d*(1.0 - v + maths.log(v))) then
				return scale*d*v
			end
		end
	end
	return {
		sample = sample,
		logprob = terra(x: real, shape: real, scale: real) : real
			if x <= 0.0 then return [-math.huge] end
			return (shape - 1.0)*maths.log(x) - x/scale - [log_gamma(real)](shape) - shape*maths.log(scale)
		end
	}
end)

--------------------------------------------

local log_beta = S.memoize(function(real)
	local lg = log_gamma(real)
	return terra(a: real, b: real)
		return lg(a) + lg(b) - lg(a+b)
	end
end)

D.beta = S.memoize(function(real)
	return {
		sample = terra(a: real, b: real) : real
			var x = [D.gamma(real)].sample(a, 1.0)
			return x / (x + [D.gamma(real)].sample(b, 1.0))
		end,
		logprob = terra(x: real, a: real, b: real) : real
			if x > 0.0 and x < 1.0 then
				return (a-1.0)*maths.log(x) + (b-1.0)*maths.log(1.0-x) - [log_beta(real)](a,b)
			else
				return [-math.huge]
			end
		end
	}
end)

--------------------------------------------

local g = S.memoize(function(real)
	return terra(x: real)
		if x == 0.0 then return 1.0 end
		if x == 1.0 then return 0.0 end
		var d = 1.0 - x
		return (1.0 - (x * x) + (2.0 * x * maths.log(x))) / (d * d)
	end
end)

D.binomial = S.memoize(function(real)
	local inv2 = 1/2
	local inv3 = 1/3
	local inv6 = 1/6
	local flt = double
	return {
		sample = terra(p: real, n: int) : int
			var k = 0
			var N = 10
			var a:flt, b:flt
			while n > N do
				a = 1 + n/2
				b = 1 + n-a
				var x = [D.beta(flt)].sample(a, b)
				if x >= p then
					n = a - 1
					p = p / x
				else
					k = k + a
					n = b - 1
					p = (p-x) / (1.0 - x)
				end
			end
			var u:flt
			for i=0,n do
				u = R.random()
				if u < p then k = k + 1 end
			end
			return k
		end,
		logprob = terra(s: int, p: real, n: int) : real
			if s < 0 or s >= n then return [-math.huge] end
			var q = 1.0-p
			var S = s + inv2
			var T = n - s - inv2
			var d1 = s + inv6 - (n + inv3) * p
			var d2 = q/(s+inv2) - p/(T+inv2) + (q-inv2)/(n+1)
			d2 = d1 + 0.02*d2
			var num = 1.0 + q * [g(real)](S/(n*p)) + p * [g(real)](T/(n*q))
			var den = (n + inv6) * p * q
			var z = num / den
			var invsd = maths.sqrt(z)
			z = d2 * invsd
			return [D.gaussian(real)].logprob(z, 0.0, 1.0) + maths.log(invsd)
		end
	}
end)

--------------------------------------------

local terra fact(x: int)
	var t:int = 1
	while x > 1 do
		t = t * x
		x = x - 1
	end
	return t	
end

local terra lnfact(x: int)
	if x < 1 then x = 1 end
	if x < 12 then return maths.log(fact(x)) end
	var invx = 1.0 / x
	var invx2 = invx*invx
	var invx3 = invx2*invx
	var invx5 = invx3*invx2
	var invx7 = invx5*invx2
	var ssum = ((x + 0.5) * maths.log(x)) - x
	ssum = ssum + maths.log(2*[math.pi]) / 2.0
	ssum = ssum + (invx / 12) - (invx / 360)
	ssum = ssum + (invx5 / 1260) - (invx7 / 1680)
	return ssum
end

D.poisson = S.memoize(function(real)
	return {
		sample = terra(mu: real) : int
			var k = 0.0
			while mu > 10 do
				var m = (7.0/8)*mu
				var x = [D.gamma(real)].sample(m, 1.0)
				if x > mu then
					return int(k + [D.binomial(real)].sample(mu/x, (m-1)))
				else
					mu = mu - x
					k = k + m
				end
			end
			var emu = maths.exp(-mu)
			var p = 1.0
			while p > emu do
				p = p * R.random()
				k = k + 1
			end
			return int(k-1)
		end,
		logprob = terra(k: int, mu: real) : real
			if k < 0 then return [-math.huge] end
			return k * maths.log(mu) - mu - lnfact(k)
		end
	}
end)

--------------------------------------------

D.categorical = S.memoize(function(real)
	return {
		sample = terra(params: &S.Vector(real)) : int
			var sum = real(0.0)
			for i=0,params:size() do sum = sum + params(i) end
			var result: int = 0
			var x = R.random() * sum
			var probAccum = real(0.0)
			repeat
				probAccum = probAccum + params(result)
				result = result + 1
			until probAccum > x or result == params:size()
			return result - 1
		end,
		logprob = terra(val: int, params: &S.Vector(real)) : real
			if val < 0 or val >= params:size() then
				return [-math.huge]
			end
			var sum = real(0.0)
			for i=0,params:size() do
				sum = sum + params(i)
			end
			return maths.log(params(val)/sum)
		end
	}
end)

--------------------------------------------

D.dirichlet = S.memoize(function(real)
	return {
		-- NOTE: It is up to the caller to manage the memory of the
		--    returned vector.
		sample = terra(params: &S.Vector(real)) : S.Vector(real)
			var out : S.Vector(real)
			out:init(params:size())
			for i=0,params:size() do out:insert() end
			var ssum = real(0.0)
			for i=0,params:size() do
				var t = [D.gamma(real)].sample(params(i), 1.0)
				out(i) = t
				ssum = ssum + t
			end
			for i=0,params:size() do
				out(i) = out(i)/ssum
			end
			return out
		end,
		logprob = terra(theta: &S.Vector(real), params: &S.Vector(real)) : real
			var sum = real(0.0)
			for i=0,params:size() do sum = sum + params(i) end
			var logp = [log_gamma(real)](sum)
			for i=0,params:size() do
				var a = params(i)
				logp = logp + (a - 1.0)*maths.log(theta(i))
				logp = logp - [log_gamma(real)](a)
			end
			return logp
		end
	}
end)

--------------------------------------------

return D

end)
