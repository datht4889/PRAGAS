import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares  

from .min_norm_solvers import MinNormSolver, gradient_normalizers


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        try:
            loss.backward()
            return loss, extra_outputs
        except:
            print("MOO failed")
            return None

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class NashMTL(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super().__init__(
            n_tasks=n_tasks,
            device=device,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                    )
                )
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha
        #alpha = alpha / (torch.sum(alpha**2)**(1/2)+1e-6)
        #print(alpha, losses)
        #alpha = [i if losses[idx] > -np.log(0.99) else min(i, 1) for idx, i in enumerate(alpha)]
        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        #mags = [abs(i.item())+1e-8 for i in losses]
        #losses = [i/mags[idx] for idx, i in enumerate(losses)]
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )
        loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss, extra_outputs

class FairGrad(WeightMethod):
    """
    FairGrad.

    This method is proposed in `Fair Resource Allocation in Multi-Task Learning (ICML 2024)
    <https://openreview.net/forum?id=KLmWRMg6nL>`_ and implemented by modifying from the 
    `official PyTorch implementation <https://github.com/OptMN-Lab/fairgrad>`_.
    """
    def __init__(self, n_tasks: int, device: torch.device, **kwargs):
        super().__init__(n_tasks, device, **kwargs)
        self.kwargs = kwargs
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the weighted loss using FairGrad.

        Parameters
        ----------
        losses : torch.Tensor
            Tensor of task-specific losses.
        shared_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        task_specific_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        last_shared_parameters : Union[List[torch.nn.parameter.Parameter], torch.Tensor]
        representation : Union[torch.nn.parameter.Parameter, torch.Tensor]
        kwargs : dict
            Additional arguments, including FairGrad_alpha.

        Returns
        -------
        loss : torch.Tensor
            Weighted loss for backward computation.
        extra_outputs : dict
            Additional outputs, including task weights.
        """
        alpha = self.kwargs.get("FairGrad_alpha", 0.5)

        if representation is not None:
            raise ValueError("FairGrad does not support representation gradients (rep_grad=True)")

        # Compute gradient matrix G
        grads = self._compute_grad(losses, shared_parameters)
        GTG = torch.mm(grads, grads.t())

        # Solve the FairGrad optimization problem using Torch
        x = torch.ones(self.n_tasks, device=self.device, requires_grad=True) / self.n_tasks
        A = GTG.data.cpu().numpy()

        def objfn(x):
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x.cpu().detach().numpy(), bounds=(0, np.inf))
        w_cpu = res.x


        weights = torch.Tensor(w_cpu).to(self.device)

        # Compute weighted loss
        try:
            weighted_loss = sum([losses[i] * weights[i] for i in range(len(weights))])
        except:
            print("FairGrad failed")
            print(weights.shape)
            print(losses)
        # weighted_loss = torch.sum(weights * losses)

        # Return loss and extra outputs
        extra_outputs = {"weights": weights.cpu().numpy()}
        return weighted_loss, extra_outputs

    def _compute_grad(self, losses: torch.Tensor, parameters: List[torch.nn.parameter.Parameter]):
        """
        Compute gradients of the losses w.r.t. the shared parameters.

        Parameters
        ----------
        losses : torch.Tensor
        parameters : List[torch.nn.parameter.Parameter]

        Returns
        -------
        grads : torch.Tensor
            Matrix of gradients (n_tasks x n_parameters).
        """
        grads = []
        for i in range(self.n_tasks):
            grad = torch.autograd.grad(
                losses[i], parameters, retain_graph=True, create_graph=False, allow_unused=True
            )
            grad = [
                g.flatten() if g is not None else torch.zeros_like(p, device=self.device).flatten()
                for g, p in zip(grad, parameters)
            ]
            grad_flat = torch.cat(grad)
            grads.append(grad_flat)

        return torch.stack(grads)

class ExcessMTL(WeightMethod):
    r"""ExcessMTL.

    This method is proposed in `Robust Multi-Task Learning with Excess Risks (ICML 2024) <https://openreview.net/forum?id=JzWFmMySpn>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/yifei-he/ExcessMTL/blob/main/LibMTL/LibMTL/weighting/ExcessMTL.py>`_. 

    """

    def __init__(
            self, 
            n_tasks: int, 
            device: torch.device, 
            max_norm: float = 1.0,
            **kwargs):
        
        super().__init__(
            n_tasks=n_tasks, 
            device=device)
        
        self.max_norm = max_norm
        
        self.loss_weight = torch.ones(n_tasks, device=device, requires_grad=False)
        self.grad_sum = None
        self.first_epoch = True
        self.kwargs = kwargs
        self.robust_step_size = kwargs.get("robust_step_size", 1e-4)
        self.rep_grad = kwargs.get("rep_grad", False)

    def _compute_grad(self, losses: torch.Tensor, parameters: List[torch.nn.parameter.Parameter]):
        """
        Compute gradients of the losses w.r.t. the shared parameters.

        Parameters
        ----------
        losses : torch.Tensor
        parameters : List[torch.nn.parameter.Parameter]

        Returns
        -------
        grads : torch.Tensor
            Matrix of gradients (n_tasks x n_parameters).
        """
        grads = {}
        for i in range(self.n_tasks):
            grad = list(torch.autograd.grad(
                    losses[i], 
                    parameters, 
                    retain_graph=True,
                    allow_unused=True
                ))

            grad = [
                g.flatten() if g is not None and not torch.isnan(g).any() else torch.zeros_like(p, device=self.device).flatten()
                for g, p in zip(grad, parameters)
            ]
            grad_flat = torch.cat(grad)
            # grad_flat = torch.cat([torch.flatten(g) for g in grad])
            if torch.isnan(grad_flat).any():
                raise ValueError("Gradient value contains NaN.")
            grads[i] = grad_flat
        return torch.stack(tuple(v for v in grads.values()))
        

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        # Compute gradients
        shared_grads = self._compute_grad(losses, shared_parameters)

        # if self.rep_grad:
        #     per_grads, shared_grads = grads[0], grads[1]
        #     grads = []
        #     for grad in per_grads:
        #         grads.append(torch.sum(grad, dim=0))
        #     grads = torch.stack(grads)
        # else:
        #     shared_grads = grads

        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(shared_grads)

        w = torch.zeros(self.n_tasks, device=self.device)
        for i in range(self.n_tasks):
            self.grad_sum[i] += shared_grads[i] ** 2
            grad_i = shared_grads[i]
            # h_i = torch.sqrt(self.grad_sum[i] + 1e-7)
            h_i = torch.sqrt(self.grad_sum[i] + 1e-5)
            w[i] = grad_i * (1 / h_i) @ grad_i.t()
            

        if self.first_epoch:
            self.initial_w = w
            self.first_epoch = False
        else:
            # w = w / (self.initial_w + 1e-4)
            w = w / torch.clamp(self.initial_w, min=1e-2)
            self.loss_weight = self.loss_weight * torch.exp(w * self.robust_step_size)
            self.loss_weight = self.loss_weight.detach().clone()
            
            # log sum exp trick
            # 1
            # self.loss_weight = self.loss_weight / self.loss_weight.sum() * self.n_tasks

            # 2
            log_loss_weight = torch.log(self.loss_weight)
            max_log_loss_weight = torch.max(log_loss_weight)
            log_sum = max_log_loss_weight + torch.log(torch.exp(log_loss_weight - max_log_loss_weight).sum())
            self.loss_weight = torch.exp(log_loss_weight - log_sum)
            self.loss_weight = self.loss_weight.detach().clone()
            if self.loss_weight.isnan().any():
                print("Loss weight: ", self.loss_weight)
                print("Loss weight sum: ", self.loss_weight.sum())
                raise ValueError("loss_weight contains NaN in weighted_methods.py (1)")
            self.loss_weight = self.loss_weight * self.n_tasks
            ### done ###

            if self.loss_weight.isnan().any():
                print("Loss weight: ", self.loss_weight)
                print("Loss weight sum: ", self.loss_weight.sum())
                raise ValueError("loss_weight contains NaN in weighted_methods.py (2)")
            self.loss_weight = self.loss_weight.detach().clone()

        # Compute weighted loss
        try:
            # loss = torch.mul(torch.Tensor(losses).to(self.device), self.loss_weight).sum()
            loss = sum([losses[i] * self.loss_weight[i] for i in range(len(self.loss_weight))])
            extra_outputs = {'loss_weights': self.loss_weight.cpu().detach().numpy()}
            return loss, extra_outputs
        except:
            print(losses)
            raise ValueError("ExcessMTL failed")
        
    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        #mags = [abs(i.item())+1e-8 for i in losses]
        #losses = [i/mags[idx] for idx, i in enumerate(losses)]
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )
        loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, 
                                           self.max_norm, 
                                           error_if_nonfinite=True
                                           )

        return loss, extra_outputs

class FAMO(WeightMethod):
    """
    Fast Adaptive Multitask Optimization.
    https://github.com/Cranial-XIX/FAMO/blob/main/famo.py
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,   # the regularization coefficient
        w_lr: float = 0.001,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        super().__init__(
            n_tasks=n_tasks, 
            device=device)
                
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = torch.stack(losses)
        z = F.softmax(self.w, -1)
        D = torch.stack(losses) - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        extra_outputs = {"weights": z, "logits": self.w.detach().clone()}
        return loss, extra_outputs

    def update(self, curr_loss):
        delta = (self.prev_loss             - self.min_losses + 1e-8).log() - \
                (torch.stack(curr_loss)     - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
    ) -> Union[torch.Tensor, None]:
        """
        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        ----------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss, extra_outputs



class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)


class MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none"
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        # Our code
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class PCGrad(WeightMethod):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(self, losses, shared_parameters, task_specific_parameters=None):
        # shared part
        shared_grads = []
        for l in losses:
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [
                        torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                        for grad_i, grad_j in zip(g_i, g_j)
                    ]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    )
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        return torch.tensor(1.123), {}
        return None, {}  # NOTE: to align with all other weight methods


class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4):
        super().__init__(n_tasks, device=device)
        self.c = c

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        self.get_weighted_loss(losses, shared_parameters)
        return None, {}  # NOTE: to align with all other weight methods


class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)


class IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        D = (
            G[
                0,
            ]
            - G[
                1:,
            ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
            U[
                0,
            ]
            - U[
                1:,
            ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum(losses * alpha)

        return loss, dict(weights=alpha)


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    uw=Uncertainty,
    pcgrad=PCGrad,
    mgda=MGDA,
    cagrad=CAGrad,
    nashmtl=NashMTL,
    scaleinvls=ScaleInvariantLinearScalarization,
    rlw=RLW,
    imtl=IMTLG,
    dwa=DynamicWeightAverage,
)
