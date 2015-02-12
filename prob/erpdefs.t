return require("platform.module")(function(platform)

local erp = require("prob.erp")(platform)
local distrib = require("prob.distrib")(platform)


local ERPs = {}

-------------------------------------------------------------------------------

ERPs.flip = erp.makeERP(
	distrib.bernoulli(double).sample,
	distrib.bernoulli(double).logprob,
	-- Deterministic flip proposal
	terra(currval: bool, p: double)
		return (not currval), 0.0, 0.0
	end
)

-------------------------------------------------------------------------------

ERPs.uniform = erp.makeERP(
	distrib.uniform(double).sample,
	distrib.uniform(double).logprob
)

-------------------------------------------------------------------------------

ERPs.gaussian = erp.makeERP(
	distrib.gaussian(double).sample,
	distrib.gaussian(double).logprob,
	-- Drift proposal
	terra(currval: double, mu: double, sigma: double)
		var newval = [distrib.gaussian(double)].sample(currval, sigma)
		var fwdlp = [distrib.gaussian(double)].sample(newval, currval, sigma)
		var rvslp = [distrib.gaussian(double)].sample(currval, newval, sigma)
		return newval, fwdlp, rvslp
	end
)

-------------------------------------------------------------------------------

ERPs.gamma = erp.makeERP(
	distrib.gamma(double).sample,
	distrib.gamma(double).logprob
)

-------------------------------------------------------------------------------

ERPs.beta = erp.makeERP(
	distrib.beta(double).sample,
	distrib.beta(double).logprob
)

-------------------------------------------------------------------------------

ERPs.binomial = erp.makeERP(
	distrib.binomial(double).sample,
	distrib.binomial(double).logprob
)

-------------------------------------------------------------------------------

ERPs.categorical = erp.makeERP(
	distrib.categorical(double).sample,
	distrib.categorical(double).logprob
)

-------------------------------------------------------------------------------

ERPs.dirichlet = erp.makeERP(
	distrib.dirichlet(double).sample,
	distrib.dirichlet(double).logprob
)

-------------------------------------------------------------------------------

return ERPs

end)