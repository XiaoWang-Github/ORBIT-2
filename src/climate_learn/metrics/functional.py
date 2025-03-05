# Standard library
from typing import Optional, Union, List, Dict

from sympy import true

# Local application
from .utils import Pred, handles_probabilistic

# Third party
import torch
import torch.nn.functional as F
import lpips
from torchmetrics.functional.image import image_gradients
from einops import repeat
import torchvision

@handles_probabilistic
def perceptual(
    loss_fn,
    model,
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor]
) -> Union[torch.FloatTensor, torch.DoubleTensor]:

#    print("loss_fn(pred,target).shape",temp.shape,"max",torch.max(temp),"min",torch.min(temp),flush=True)

#    if torch.distributed.get_rank()==0:
#        torchvision.utils.save_image(temp[0],'temp.png')


    error = F.l1_loss(pred, target) + 0.5*torch.mean(loss_fn(pred,target))

    return error

@handles_probabilistic
def lat_weighted_quantile(
        pred: Pred, 
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        aggregate_only: bool = False,
        lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None, 
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    """ Latitude weighted quantile loss """

    # -3s to 3s
    QUANTILES = [1 - 0.9987, 1 - 0.9772, 1 - 0.8413, 0.5000, 0.8413, 0.9772, 0.9987]
    #QUANTILES_LABELS = ["-3s", "-2s", "-1s", "0s", "+1s", "+2s", "+3s"]
    quantiles_tensor = torch.tensor(QUANTILES, device=pred.device)

    error = pred -  target # [N, C, H, W]
    # latitude weights
    if lat_weights is not None:
        error = error * lat_weights
    error = error.unsqueeze(-1).expand(-1, -1, -1, -1, len(QUANTILES))
    losses = torch.max((quantiles_tensor - 1) * error, quantiles_tensor * error) # (B, V, H, W, Q)
    loss = torch.abs(losses).mean() 
    return loss

@handles_probabilistic
def image_gradient(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    """
    Computes the image gradient loss between the ground truth and predicted images.

    Args:
        target (torch.Tensor): Ground truth image tensor of shape (B, C, V, H, W) or (B, V, H, W).
        pred (torch.Tensor): Predicted image tensor with the same shape as `image`.
    Returns:
        dict: A dictionary containing the gradient loss.
    """

    mse_error =  torch.mean((pred - target).square())
    loss = mse_error + .1 *torch.mean(image_gradient_fn(pred, target))
    return loss

@handles_probabilistic
def image_gradient_fn(pred:Pred, 
                      target: Union[torch.BFloat16Tensor, torch.FloatTensor, torch.DoubleTensor]
                      ):
    
    # Ensure images are at least 4D (batch, V, H, W)
    if pred.dim() == 5:
        pred = pred.flatten(0, 1)  # Merge batch and channel dimensions

    if target.dim() == 5:
        target = target.flatten(0, 1)

    # Compute image gradients
    dy, dx = image_gradients(target)
    hat_dy, hat_dx = image_gradients(pred)

    # Compute gradient difference loss: add latitude weight if needed
    error = torch.mean(torch.abs(dx - hat_dx) + torch.abs(dy - hat_dy))
    return error

@handles_probabilistic
def mse(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    var_names: Optional[List[str]] = None,
    var_weights: Optional[Dict[str, float]] = None,
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:

    error = (pred - target).square()
    #print('during  mse with error', error.dtype, pred.dtype, target.dtype)
    if lat_weights is not None:
        error = error * lat_weights

    if var_names is not None:
        assert len(var_names) == pred.shape[1], "Number of variable names must match channel dimension"

        channel_weights = torch.ones(pred.shape[1], device=pred.device, dtype=pred.dtype)
        for i, var in enumerate(var_names):
            weight = var_weights.get(var, 1.0)
            channel_weights[i] = weight
        weights_expanded = channel_weights.view(1, -1, 1, 1)
        error = error * weights_expanded

    per_channel_losses = error.mean([0, 2, 3])
    loss = error.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def msess(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred_mse = mse(pred, target, aggregate_only, lat_weights)
    clim_mse = mse(climatology, target, aggregate_only, lat_weights)
    return 1 - pred_mse / clim_mse


@handles_probabilistic
def mae(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).abs()
    if lat_weights is not None:
        error = error * lat_weights
    per_channel_losses = error.mean([0, 2, 3])
    loss = error.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def rmse(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
    mask=None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    if mask is not None:
        error = error * mask
        eps = 1e-9
        masked_lat_weights = torch.mean(mask, dim=(1, 2, 3), keepdim=True) + eps
        error = error / masked_lat_weights
    per_channel_losses = error.mean([2, 3]).sqrt().mean(0)
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


@handles_probabilistic
def acc(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Optional[Union[torch.FloatTensor, torch.DoubleTensor]],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
    mask=None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = pred - climatology
    target = target - climatology
    per_channel_acc = []
    for i in range(pred.shape[1]):
        pred_prime = pred[:, i] - pred[:, i].mean()
        target_prime = target[:, i] - target[:, i].mean()
        if mask is not None:
            eps = 1e-9
            numer = (mask * lat_weights * pred_prime * target_prime).sum()
            denom1 = ((mask + eps) * lat_weights * pred_prime.square()).sum()
            denom2 = ((mask + eps) * lat_weights * target_prime.square()).sum()
        else:
            numer = (lat_weights * pred_prime * target_prime).sum()
            denom1 = (lat_weights * pred_prime.square()).sum()
            denom2 = (lat_weights * target_prime.square()).sum()
        numer = (lat_weights * pred_prime * target_prime).sum()
        denom1 = (lat_weights * pred_prime.square()).sum()
        denom2 = (lat_weights * target_prime.square()).sum()
        per_channel_acc.append(numer / (denom1 * denom2).sqrt())
    per_channel_acc = torch.stack(per_channel_acc)
    result = per_channel_acc.mean()
    if aggregate_only:
        return result
    return torch.cat((per_channel_acc, result.unsqueeze(0)))


@handles_probabilistic
def pearson(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = _flatten_channel_wise(pred)
    target = _flatten_channel_wise(target)
    pred = pred - pred.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    per_channel_coeffs = F.cosine_similarity(pred, target)
    coeff = torch.mean(per_channel_coeffs)
    if not aggregate_only:
        coeff = coeff.unsqueeze(0)
        coeff = torch.cat((per_channel_coeffs, coeff))
    return coeff


@handles_probabilistic
def mean_bias(
    pred: Pred,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    per_channel_mb = []
    for i in range(pred.shape[1]):
        per_channel_mb.append(target[:, i].mean() - pred[:, i].mean())
    per_channel_mb = torch.stack(per_channel_mb)
    result = per_channel_mb.mean()
    if aggregate_only:
        return result
    return torch.cat((per_channel_mb, result.unsqueeze(0)))


def _flatten_channel_wise(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: A tensor of shape [B,C,H,W].
    :type x: torch.Tensor

    :return: A tensor of shape [C,B*H*W].
    :rtype: torch.Tensor
    """
    subtensors = torch.tensor_split(x, x.shape[1], 1)
    result = torch.stack([t.flatten() for t in subtensors])
    return result


def gaussian_crps(
    pred: torch.distributions.Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    mean, std = pred.loc, pred.scale
    z = (target - mean) / std
    standard_normal = torch.distributions.Normal(
        torch.zeros_like(pred), torch.ones_like(pred)
    )
    pdf = torch.exp(standard_normal.log_prob(z))
    cdf = standard_normal.cdf(z)
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / torch.pi)
    if lat_weights is not None:
        crps = crps * lat_weights
    per_channel_losses = crps.mean([0, 2, 3])
    loss = crps.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def gaussian_spread(
    pred: torch.distributions.Normal,
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    variance = torch.square(pred.scale)
    if lat_weights is not None:
        variance = variance * lat_weights
    per_channel_losses = variance.mean([2, 3]).sqrt().mean(0)
    loss = variance.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def gaussian_spread_skill_ratio(
    pred: torch.distributions.Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    spread = gaussian_spread(pred, aggregate_only, lat_weights)
    error = rmse(pred, target, aggregate_only, lat_weights)
    return spread / error


def nrmses(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    clim: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    y_normalization = clim.squeeze()
    error = (pred.mean(dim=0) - target.mean(dim=0)) ** 2  # (C, H, W)
    if lat_weights is not None:
        error = error * lat_weights.squeeze(0)
    per_channel_losses = error.mean(dim=(-2, -1)).sqrt() / y_normalization  # C
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))


def nrmseg(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    clim: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    y_normalization = clim.squeeze()
    if lat_weights is not None:
        pred = pred * lat_weights
        target = target * lat_weights
    pred = pred.mean(dim=(-2, -1))  # N, C
    target = target.mean(dim=(-2, -1))  # N, C
    error = (pred - target) ** 2
    per_channel_losses = error.mean(0).sqrt() / y_normalization  # C
    loss = per_channel_losses.mean()
    if aggregate_only:
        return loss
    return torch.cat((per_channel_losses, loss.unsqueeze(0)))
