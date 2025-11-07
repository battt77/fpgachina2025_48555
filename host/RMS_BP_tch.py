import torch
import torch.nn as nn
from torch.autograd import Function
import numbers

# ===============================================================
# RMSNorm 的自定义前向和反向传播实现
# ===============================================================
class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, normalized_shape=12, eps=1e-6, batch_size=2):
        """
        x: 输入张量[2, 12]
        weight: 可学习缩放参数 γ [12]
        normalized_shape: 最后 D 个维度
        eps: 数值稳定项
        batch_size: 保留参数（为了兼容性，但不再使用）
        """
        # ---- 1. 计算标准 RMS（对整个最后一个维度）----
        mean_sq = torch.mean(x ** 2, dim=-1, keepdim=True)  # [2, 1]
        rms = torch.sqrt(mean_sq + eps)  # [2, 1]

        # ---- 2. 归一化 ----
        norm = x / rms  # [2, 12]

        # ---- 3. 应用 weight（γ）----
        result = norm * weight  # weight自动广播 [2, 12]

        # ---- 4. 保存变量供 backward 使用 ----
        ctx.save_for_backward(x, rms, weight)
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, rms, weight = ctx.saved_tensors
        eps = ctx.eps
        normalized_shape = ctx.normalized_shape
        
        grad_input = grad_weight = None
        
        # 计算维度
        D = normalized_shape  # 12
        inv_rms = 1.0 / rms  # [2, 1]
        print("inv_rms",inv_rms)
        norm = x * inv_rms  # [2, 12]
        
        # ---- 计算输入x梯度 ----
        if ctx.needs_input_grad[0]:
            partone = grad_output * weight * inv_rms
            parttwo = x *torch.sum(grad_output * weight * inv_rms**3 / D * x , dim=-1, keepdim=True) # [2, 2, 6]
            # grad_input = partone - parttwo # [2, 2, 6]
            grad_input = weight * inv_rms * (grad_output - 1/D *(norm * torch.sum(grad_output * norm,dim=-1, keepdim=True)))

            # grad_input = torch.reshape(grad_input_c, input_shape)

        # ---- 计算 gamma 梯度 ----
        if weight is not None and ctx.needs_input_grad[1]:
            grad_weight = torch.sum(grad_output*norm,dim=0,keepdim=True)  # [2, 2, 6]
            # 将梯度重塑回原始 weight 的形状
            # grad_weight = torch.reshape(grad_weight_reshaped, input_shape)

        return grad_input, grad_weight, None, None, None


# ===============================================================
# 模块封装
# ===============================================================
class RMSNormTrue(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None, batch_size=2):
        super(RMSNormTrue, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.batch_size = batch_size

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return RMSNormFunction.apply(x, self.weight, self.normalized_shape, self.eps, self.batch_size)

class My_net(torch.nn.Module):
  def __init__(self):
    super(My_net,self).__init__()
    self.myrms=RMSNormTrue(12,eps=1e-12,batch_size=2,elementwise_affine=True)
  def forward(self,x):
    y = self.myrms(x)
    return y


# ===============================================================
# 测试验证
# ===============================================================
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

# pytorch的输出y
  ly= torch.nn.RMSNorm(12)
  rms_y= ly(x1)
  print(rms_y)

# 自己写的输出y
  net=My_net()
  print(net)
  verify_rms_y=net(x2)
  print(verify_rms_y)
  print(rms_y/verify_rms_y)

# pytorch的
  log_probs = torch.nn.functional.log_softmax(rms_y, dim=-1)
  per_example_loss = -torch.sum( log_probs, dim=-1)
  loss = torch.mean(per_example_loss)
  loss.backward()
  print(loss)
  print(x1.grad)

# 自己写的
  log_probs2 = torch.nn.functional.log_softmax(verify_rms_y, dim=-1)
  per_example_loss2 = -torch.sum( log_probs2, dim=-1)
  loss_v = torch.mean(per_example_loss2)
  loss_v.backward()
  print(loss_v)
  print(x2.grad)

  for name, parms in ly.named_parameters():
    print('lyname:', name)
    print(parms.grad)

  for name, parms in net.named_parameters():
    print('netname:', name)
    print(parms.grad)
