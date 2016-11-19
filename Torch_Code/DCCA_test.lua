require 'cutorch'
require 'nn'
dofile 'cpu-DCCA.lua'
cmd = torch.CmdLine()

cmd:option('-rcov1', 0.01, [[rcov1 for CCA]])
cmd:option('-rcov2', 0.01, [[rcov2 for CCA]])
cmd:option('-k', 3, [[K for CCA]])

opt = cmd:parse(arg)
cutorch.setDevice(1)

cca_model = nn.DCCA(opt)
local m1 = torch.randn(3,4)
local m2 = torch.randn(3,5)
local m1c = m1:cuda()
local m2c = m2:cuda()
local corrl = cca_model:forward({m1,m2})
local dDcca = cca_model:backward({m1,m2})
print(dDcca)
