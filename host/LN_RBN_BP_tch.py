import torch 

class Function(torch.autograd.Function):

  def forward(ctx,x,weight,bias,normalized_shape=12,eps=1e-12,batch_size=2):
      
      input_shape = list(x.shape)  #获取输入形状
      bottom_dims=int(normalized_shape/batch_size)  #计算每个块的大小
      x_c=torch.reshape(x,shape=input_shape[:-1]+[batch_size,bottom_dims])
      c_x_shape=x_c.shape
      batch_size=torch.tensor(batch_size)
      C=torch.rsqrt(2*torch.log(batch_size))

      max = torch.max(x_c,dim=-1,keepdim=True).values
      min = torch.min(x_c,dim=-1,keepdim=True).values

      max_pos=torch.where(x_c==max,1,0)
      min_pos=torch.where(x_c==min,1,0)

      range_var = torch.subtract(max,min)

      range_var_sum = torch.sum(range_var,dim=-2)

      range_var_div = range_var_sum/(2*batch_size)

      variance= torch.multiply(C,range_var_div)
      
      keep_dims=True
      if keep_dims is None:
        keep_dims=False
      y = x
      mean = torch.mean(y,dim=-1,keepdim=keep_dims)
      if not keep_dims:
        mean = torch.squeeze(mean,-1)
        variance = torch.squeeze(variance,-1)

      inv=torch.reciprocal(variance+eps)

      norm=x*inv+(-mean*inv)
      gamma = torch.tensor(weight)
      beta = torch.tensor(bias)
      result = x*inv*gamma+(beta-mean*inv*gamma).requires_grad_(True)
      
      normalized_shape=torch.tensor(normalized_shape)
      batch_size=torch.tensor(batch_size)
      eps=torch.tensor(eps)
      c_x_shape=torch.tensor(c_x_shape)
      ctx.save_for_backward(normalized_shape,norm,weight,bias,variance,eps,batch_size,max_pos,min_pos,C,c_x_shape)
      return result
  
  def backward(ctx,grad_output):
        grad_input=grad_weight=grad_bias=grad_normalized_shape=grad_eps=grad_batch_size=None
        normalized_shape,norm,weight,bias,variance,eps,batch_size,max_pos,min_pos,C,c_x_shape=ctx.saved_tensors
        var=torch.add(variance,eps)
        gamma = weight
        norm=norm
        c_x_shape=list(c_x_shape)
        H=(1/normalized_shape)                       #1/H
        C=C/(2*batch_size)      #C/2n
        if ctx.needs_input_grad[0]:
          partone = torch.reciprocal(var)*gamma*grad_output
          parttwo = torch.reciprocal(var)*gamma*H*torch.sum(grad_output, dim=-1, keepdim=True)
          partthree = torch.reciprocal(var)*gamma*C*torch.sum(torch.multiply(grad_output,norm),dim=-1,keepdim=True)

          x_delta =  partone-parttwo        #others
          x_delta_origin_shape = x_delta.shape

          x_delta = torch.reshape(x_delta,shape=c_x_shape)
          partthree = torch.reshape(partthree,shape=c_x_shape)
          
          #max_pos=torch.squeeze(max_pos,dim=-1)
          #min_pos=torch.squeeze(min_pos,dim=-1)

          #max_one_hot=torch.nn.functional.one_hot(max_pos,num_classes=c_x_shape[-1])       #the max num position
          #min_one_hot=torch.nn.functional.one_hot(min_pos,num_classes=c_x_shape[-1])       #the min num position
        
          x_delta = x_delta-partthree*max_pos+partthree*min_pos
          grad_input = torch.reshape(x_delta,x_delta_origin_shape)
        if weight is not None and ctx.needs_input_grad[1]:
          grad_weight = torch.sum(norm*grad_output,dim=0,keepdim=True)
        if bias is not None and ctx.needs_input_grad[2]: 
          grad_bias = torch.sum(grad_output, dim=0,keepdim=True)
        return grad_input,grad_weight,grad_bias,grad_normalized_shape,grad_eps,grad_batch_size    


class LayerNormR(torch.nn.Module):
    def __init__(self,normalized_shape,elementwise_affine=True,eps=1e-5,bias=True,device=None,dtype=None,batch_size=8):
      super(LayerNormR,self).__init__()
      self.epsilon=eps
      self.mean = None
      self.variance = None
      factory_kwargs = {'device': device, 'dtype': dtype}
      self.normalized_shape = normalized_shape
      self.elementwise_affine = elementwise_affine
      self.batch_size=batch_size
      
      if self.elementwise_affine:
        self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        if bias:
          self.bias = torch.nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
          self.register_parameter('bias',None)
      else:
        self.register_parameter('weight',None)
        self.register_parameter('bias',None)
        
      self.reset_parameters()
    
    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
    
    def forward(self,x):
      result=Function.apply(x,self.weight,self.bias,self.normalized_shape,self.epsilon,self.batch_size)
      return result

class My_net(torch.nn.Module):
  def __init__(self):
    super(My_net,self).__init__()
    self.myln=LayerNormR(12,eps=1e-12,batch_size=2)
  def forward(self,x):
    y = self.myln(x)
    return y
  
'''simple test code
x1 = torch.tensor(
        [[18.369314, 2.6570225, 25.402943,
          10.403599, 2.7813416, 20.794857,
        19.0327, 2.6398268, 6.3894367,
          3.921237, 10.761424, 2.7887821],
        [11.466338, 20.210938, 8.242946,
          22.77081, 11.555874, 11.183836,
        8.976935, 10.204252, 11.20231,
          -7.356888, 6.2725096, 1.1952505]],requires_grad=True)
weight = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1],requires_grad=True,dtype=torch.float32)
bias = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0],requires_grad=True,dtype=torch.float32)
y=Function.apply(x1,weight,bias)
log_probs = torch.nn.functional.log_softmax(y, dim=-1)
per_example_loss = -torch.sum( log_probs, dim=-1)
loss = torch.mean(per_example_loss)
loss.backward()
print(x1.grad)
'''
if __name__ =='__main__':

  x1 = torch.tensor(
        [[18.369314, 2.6570225, 25.402943,
          10.403599, 2.7813416, 20.794857,
        19.0327, 2.6398268, 6.3894367,
          3.921237, 10.761424, 2.7887821],
        [11.466338, 20.210938, 8.242946,
          22.77081, 11.555874, 11.183836,
        8.976935, 10.204252, 11.20231,
          -7.356888, 6.2725096, 1.1952505]],requires_grad=True)
  x2 = torch.tensor(
        [[18.369314, 2.6570225, 25.402943,
          10.403599, 2.7813416, 20.794857,
        19.0327, 2.6398268, 6.3894367,
          3.921237, 10.761424, 2.7887821],
        [11.466338, 20.210938, 8.242946,
          22.77081, 11.555874, 11.183836,
        8.976935, 10.204252, 11.20231,
          -7.356888, 6.2725096, 1.1952505]],requires_grad=True)

  ly= torch.nn.LayerNorm(12)
  ln_y= ly(x1)
  print(ln_y)

  net=My_net()
  print(net)
  verify_ln_y=net(x2)
  print(verify_ln_y)
  print(ln_y/verify_ln_y)

  log_probs = torch.nn.functional.log_softmax(ln_y, dim=-1)
  per_example_loss = -torch.sum( log_probs, dim=-1)
  loss = torch.mean(per_example_loss)
  loss.backward()

  log_probs2 = torch.nn.functional.log_softmax(verify_ln_y, dim=-1)
  per_example_loss2 = -torch.sum( log_probs2, dim=-1)
  loss_v = torch.mean(per_example_loss2)
  loss_v.backward()
  print(loss_v)
  print(x2.grad)

  for name, parms in ly.named_parameters():
    print(name)
    print(parms.grad)

  for name, parms in net.named_parameters():
    print(name)
    print(parms.grad)

