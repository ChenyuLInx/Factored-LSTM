local DCCA, parent = torch.class('nn.DCCA','nn.Criterion')

local eps = 1e-12

function DCCA:__init(opt)
	parent.__init(self)
	self.rcov1	= opt.rcov1 or 0
	self.rcov2	= opt.rcov2 or 0
	self.k		= opt.k
	self.gradInput = {torch.CudaTensor(), torch.CudaTensor()}
end

function DCCA:updateOutput(input, target)
	self.n  = input[1]:size(1)
	self.d1	= input[1]:size(2)
	self.d2	= input[2]:size(2)
	self.input1 = input[1] - input[1]:mean(1):expand(self.n, self.d1)
	self.input2 = input[2] - input[2]:mean(1):expand(self.n, self.d2)

	self.s11    = (self.input1:t() * self.input1)/(self.n-1)+torch.eye(self.d1):cuda()*self.rcov1
	self.s22    = (self.input2:t() * self.input2)/(self.n-1)+torch.eye(self.d2):cuda()*self.rcov2
	self.s12	= (self.input1:t() * self.input2)/(self.n-1)
	
	self.e1, self.v1 = torch.eig(self.s11, 'V')
	self.e2, self.v2 = torch.eig(self.s22, 'V')
	self.e1	= self.e1:t()[1]
	self.e2	= self.e2:t()[1]
	
	-- numerical checking 
	self.idx1	= torch.ones(self.d1):cumsum():cuda()[self.e1:ge(1e-12)]
	self.e1 = self.e1:index(1, self.idx1)
	self.v1	= self.v1:index(2, self.idx1)
	self.idx2	= torch.ones(self.d2):cumsum():cuda()[self.e2:ge(1e-12)]
	self.e2 = self.e2:index(1, self.idx2)
	self.v2	= self.v2:index(2, self.idx2)

	self.n1 = self.e1:size(1)
	self.n2 = self.e2:size(1)
	self.k11 = self.v1 * torch.cmul(torch.pow(self.e1, -0.5):reshape(self.n1,1):expand(self.n1, self.n1), torch
.eye(self.n1):cuda()) * self.v1:t()
	self.k22 = self.v2 * torch.cmul(torch.pow(self.e2, -0.5):reshape(self.n2,1):expand(self.n2, self.n2), torch
.eye(self.n2):cuda()) * self.v2:t()

	self.T = self.k11 * self.s12 * self.k22
	self.U, self.D, self.V = torch.svd(self.T)
	
	self.U = self.U:t()[{{1, self.k}}]:t()
	self.D = self.D[{{1, self.k}}]; 
	self.D = torch.cmul(self.D:reshape(self.k, 1):expand(self.k, self.k), torch.eye(self.k):cuda())
	self.V = self.V:t()[{{1, self.k}}]:t()
	self.output	= self.D:sum()
	return -self.output
end

function DCCA:updateGradInput(input, output)
	self.delta12	= (self.k11 * self.U) * (self.V:t() * self.k22)
	self.delta11	= -0.5 * (self.k11*self.U) * self.D * (self.U:t()*self.k11);
	self.delta22	= -0.5 * (self.k22*self.V) * self.D * (self.V:t()*self.k22);
	local grad1	    = 2*self.input1*self.delta11 + self.input2*self.delta12:t(); grad1 = grad1/(self.n - 1)
	self.gradInput[1]:resize(self.input1:size()):copy(grad1)
	local grad2		= self.input1*self.delta12 + 2*self.input2*self.delta22; grad2 = grad2 / (self.n - 1)
	self.gradInput[2]:resize(self.input2:size()):copy(grad2)
	return self.gradInput
end
