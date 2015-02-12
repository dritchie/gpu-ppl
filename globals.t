
-- The platform being used for this program
local platform = require("platforms.x86")
local function getPlatform() return platform end


return
{
	platform = getPlatform
}

