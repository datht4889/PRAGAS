import torch
from collections import defaultdict
import torch.nn.functional as F
import random


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if rho is not None:
                scale = rho / (grad_norm + 1e-12)
            else:
                scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FriendlySAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):

        for group in self.param_groups:
            for p in group["params"]:      
                if p.grad is None: continue       
                grad = p.grad.clone()
                if not "momentum" in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
            
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if rho is not None:
                scale = rho / (grad_norm + 1e-12)
            else:
                scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups    


class ASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eta=0.01, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, eta=eta, **kwargs)
        super(ASAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        # print("Rho value in ASAM first step: ", rho)
        wgrads = []
        for group in self.param_groups:
            # for n, p in group["params"]:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if t_w is None:
                    t_w = torch.clone(p).detach()
                    self.state[p]["eps"] = t_w
                # if 'weight' in n:
                #     t_w[...] = p[...]
                #     t_w.abs_().add_(self.defaults["eta"])
                #     p.grad.mul_(t_w)
                wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        
        for group in self.param_groups:
            # for n, p in group["params"]:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                # if 'weight' in n:
                #     p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                if rho is not None:
                    eps.mul_(rho / wgrad_norm)
                else:
                    eps.mul_(group["rho"] / wgrad_norm)
                p.add_(eps)
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            # for n, p in group["params"]:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "ASAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def centralized_gradient(x,use_gc=True,gc_conv_only=False):
    if use_gc:
      if gc_conv_only:
        if len(list(x.size()))>3:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
      else:
        if len(list(x.size()))>1:
            x.add_(-x.mean(dim = tuple(range(1,len(list(x.size())))), keepdim = True))
    return x

class GCSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(GCSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if rho is not None:
                scale = rho / (grad_norm + 1e-12)
            else:
                scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                #GC operation
                p.grad =centralized_gradient(p.grad ,use_gc=True,gc_conv_only=False)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                #p.grad =centralized_gradient(p.grad ,use_gc=True,gc_conv_only=False)
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        #first order sum 
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if rho is not None:
                scale = rho / (grad_norm + 1e-7) / self.beta
            else:
                scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                #original sam 
                # e_w = p.grad * scale.to(p)
                #asam 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w



        if zero_grad: self.zero_grad()

    '''
    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"
    '''


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self):
        inputs,targets,loss_fct,model,defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits = model(inputs)
        loss = loss_fct(logits,targets)

        l_before = loss.clone().detach()
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()
        defined_backward(loss)

        #first step to w + e(w)
        self.first_step(True)


        with torch.no_grad():
            l_after = loss_fct(model(inputs),targets)
            instance_sharpness = l_after-l_before

            #codes for sorting 
            prob = self.gamma
            if prob >=0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff,_ = torch.topk(instance_sharpness,position)
                cutoff = cutoff[-1]

                # cutoff = 0
                #select top k% 

                indices = [instance_sharpness > cutoff] 


        # second forward-backward step
        # self.first_half()

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False



        loss = loss_fct(model(inputs[indices]), targets[indices])
        loss = loss.mean()
        defined_backward(loss)
        self.second_step(True)

        self.returnthings = (predictions,return_loss)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
class LookbehindASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eta=0.01, k_steps=5, alpha=0.5, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.rho = rho
        self.eta = eta
        self.k_steps = k_steps
        self.alpha = alpha
        self.k = 0

        defaults = dict(rho=rho, eta=eta, **kwargs)
        super(LookbehindASAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.state = defaultdict(dict)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                param_state['cached_slow_params'] = torch.zeros_like(p.data)
                param_state['cached_slow_params'].copy_(p.data)
                if self.alpha == -1:
                    param_state['first_descent_step'] = torch.zeros_like(p.data)


    def get_current_k(self):
        return self.k

    def _cache_params(self):
        """ Cache the current optimizer parameters
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'].copy_(p.data)

    def _cache_slow_params(self):
        """ Cache the slow optimizer parameters
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_slow_params'].copy_(p.data)

    def _backup_and_load_slow_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_slow_params'])

    def _backup_and_load_cache(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        wgrads = []
        for group in self.param_groups:
            # for n, p in group["params"]:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if t_w is None:
                    t_w = torch.clone(p).detach()
                    self.state[p]["eps"] = t_w
                # if 'weight' in n:
                #     t_w[...] = p[...]
                #     t_w.abs_().add_(self.defaults["eta"])
                #     p.grad.mul_(t_w)
                wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        
        for group in self.param_groups:
            # for n, p in group["params"]:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                # if 'weight' in n:
                #     p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                if rho is not None:
                    eps.mul_(rho / wgrad_norm)
                else:
                    eps.mul_(group["rho"] / wgrad_norm)
                p.add_(eps)
        
        if zero_grad:
            self.zero_grad()
    

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self._backup_and_load_cache()

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

        if self.alpha == -1 and self.k == 0: #adaptive alpha
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['first_descent_step'] = torch.zeros_like(p.data)
                    param_state['first_descent_step'].copy_(p.data)

        self._cache_params()
        self._clear_and_load_backup()

        self.k += 1

        if self.k >= self.k_steps:
            self.k = 0

            # Lookbehind and cache the current optimizer parameters
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['cached_params'])
                    if self.alpha == -1: #adaptive alpha
                        cos_sim = torch.nn.CosineSimilarity(dim=0)
                        tmp_alpha = cos_sim((param_state['first_descent_step']-param_state['cached_slow_params']).flatten(), (p.data-param_state['cached_slow_params']).flatten())
                        tmp_alpha = ((tmp_alpha+1.)/2.).item()
                        p.data.mul_(tmp_alpha).add_(param_state['cached_slow_params'], alpha=1.0 - tmp_alpha)
                    else:
                        p.data.mul_(self.alpha).add_(param_state['cached_slow_params'], alpha=1.0 - self.alpha)
                    param_state['cached_params'].copy_(p.data)
                    param_state['cached_slow_params'].copy_(p.data)


class LookbehindSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eta=0.01, k_steps=5, alpha=0.5, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.rho = rho
        self.eta = eta
        self.k_steps = k_steps
        self.alpha = alpha
        self.k = 0

        defaults = dict(rho=rho, eta=eta, **kwargs)
        super(LookbehindSAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.state = defaultdict(dict)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                param_state['cached_slow_params'] = torch.zeros_like(p.data)
                param_state['cached_slow_params'].copy_(p.data)
                if self.alpha == -1:
                    param_state['first_descent_step'] = torch.zeros_like(p.data)


    def get_current_k(self):
        return self.k

    def _cache_params(self):
        """ Cache the current optimizer parameters
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'].copy_(p.data)

    def _cache_slow_params(self):
        """ Cache the slow optimizer parameters
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_slow_params'].copy_(p.data)

    def _backup_and_load_slow_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_slow_params'])

    def _backup_and_load_cache(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']


    @torch.no_grad()
    def first_step(self, zero_grad=False, rho=None):
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                eps = self.state[p].get("eps")
                if eps is None:
                    eps = torch.clone(p).detach()
                    self.state[p]["eps"] = eps
                eps[...] = p.grad[...]
                if rho is not None:
                    eps.mul_(rho / grad_norm)
                else:
                    eps.mul_(self.rho / grad_norm)
                p.add_(eps)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self._backup_and_load_cache()

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

        if self.alpha == -1 and self.k == 0: #adaptive alpha
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['first_descent_step'] = torch.zeros_like(p.data)
                    param_state['first_descent_step'].copy_(p.data)

        self._cache_params()
        self._clear_and_load_backup()

        self.k += 1

        if self.k >= self.k_steps:
            self.k = 0

            # Lookbehind and cache the current optimizer parameters
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.copy_(param_state['cached_params'])
                    if self.alpha == -1: #adaptive alpha
                        cos_sim = torch.nn.CosineSimilarity(dim=0)
                        tmp_alpha = cos_sim((param_state['first_descent_step']-param_state['cached_slow_params']).flatten(), (p.data-param_state['cached_slow_params']).flatten())
                        tmp_alpha = ((tmp_alpha+1.)/2.).item()
                        p.data.mul_(tmp_alpha).add_(param_state['cached_slow_params'], alpha=1.0 - tmp_alpha)
                    else:
                        p.data.mul_(self.alpha).add_(param_state['cached_slow_params'], alpha=1.0 - self.alpha)
                    param_state['cached_params'].copy_(p.data)
                    param_state['cached_slow_params'].copy_(p.data)