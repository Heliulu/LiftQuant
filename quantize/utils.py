
import torch
import torch.nn as nn
from scipy import linalg
import functools

def get_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('alpha') > -1:
            params.append(m)
    return iter(params) 



class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     
      





def get_act_means(model, dataloader, num_samples, bsz, keys, attention_mask,position_embeddings):
    model.eval()
    device = next(model.parameters()).device
    act_disturb = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach().cpu()
        if name in act_disturb:
            act_disturb[name] = torch.cat((act_disturb[name], tensor.to(torch.float32).to('cpu')), dim=0)
        else:
            act_disturb[name] = tensor.to(torch.float32).to('cpu')

    def stat_input_hook(m, x, y, name):
        # 捕获输入
        
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        for key in keys:
            if isinstance(m, nn.Linear) and ((key in name)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=key))
                )

    for i in range(num_samples//bsz):
        model(dataloader[i*bsz:(i+1)*bsz].to(device), attention_mask=attention_mask,position_embeddings=position_embeddings)


    for h in hooks:
        h.remove()

    return act_disturb
