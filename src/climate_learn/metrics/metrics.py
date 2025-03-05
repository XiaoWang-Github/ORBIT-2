# Standard Library
from typing import Callable, Optional, Union, List, Dict

# Local application
from .utils import MetricsMetaInfo, register
from .functional import *
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
checkpoint_wrapper,
CheckpointImpl,
apply_activation_checkpointing,
)
import functools
from torchvision.models import vgg16
import numpy as np
import torch
import os
from torch.nn import Sequential
import lpips
from lpips import NetLinLayer 
class Metric:
    """Parent class for all ClimateLearn metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        r"""
        .. highlight:: python

        :param aggregate_only: If false, returns both the aggregate and
            per-channel metrics. Otherwise, returns only the aggregate metric.
            Default is `False`.
        :type aggregate_only: bool
        :param metainfo: Optional meta-information used by some metrics.
        :type metainfo: MetricsMetaInfo|None
        """
        self.aggregate_only = aggregate_only
        self.metainfo = metainfo

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param pred: The predicted value(s).
        :type pred: torch.Tensor
        :param target: The ground truth target value(s).
        :type target: torch.Tensor

        :return: A tensor. See child classes for specifics.
        :rtype: torch.Tensor
        """
        raise NotImplementedError()


class LatitudeWeightedMetric(Metric):
    """Parent class for latitude-weighted metrics."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        lat_weights = np.cos(np.deg2rad(self.metainfo.lat))
        lat_weights = lat_weights / lat_weights.mean()
        lat_weights = torch.from_numpy(lat_weights).view(1, 1, -1, 1)
        self.lat_weights = lat_weights

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts latitude weights to the same device as `pred`.
        """
        self.lat_weights = self.lat_weights.to(device=pred.device)


class ClimatologyBasedMetric(Metric):
    """Parent class for metrics that use climatology."""

    def __init__(
        self, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None
    ):
        super().__init__(aggregate_only, metainfo)
        climatology = self.metainfo.climatology
        climatology = climatology.unsqueeze(0)
        self.climatology = climatology

    def cast_to_device(
        self, pred: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python

        Casts climatology to the same device as `pred`.
        """
        self.climatology = self.climatology.to(device=pred.device)


class TransformedMetric:
    """Class which composes a transform and a metric."""

    def __init__(self, transform: Callable, metric: Metric):
        self.transform = transform
        self.metric = metric
        self.name = metric.name

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> None:
        pred = self.transform(pred)
        target = self.transform(target)
        return self.metric(pred, target)



@register("perceptual")
class PERCEPTUAL(Metric):
    """Computes perceptual loss."""

    def __init__(self, device, model, aggregate_only: bool = False, metainfo: Optional[MetricsMetaInfo] = None):
        self.loss_fn = lpips.LPIPS(net='vgg').to(device) # best forward scores
        self.model = model

        for param in self.loss_fn.parameters():
            param.requires_grad = False


        local_rank = int(os.environ['SLURM_LOCALID'])



        #bfloat16 policy
        bfloatPolicy = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )


        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
               Sequential   # < ---- Your Transformer layer class
            },
        )


        check_fn = lambda submodule: isinstance(submodule, Sequential)



        self.loss_fn = FSDP(self.loss_fn, device_id = local_rank, process_group= None,sync_module_states=True, sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,auto_wrap_policy = None, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False)

        #activation checkpointing
        apply_activation_checkpointing(
            self.loss_fn, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
        )




        if torch.distributed.get_rank()==0:
            print("inside PERCEPTUAL after FSDP","self.loss_fn",self.loss_fn,flush=True)

        super().__init__(aggregate_only, metainfo)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return perceptual(self.loss_fn,self.model, pred, target)

@register("imagegradient")
class IMAGEGRADIENT(Metric):
    """Computes image gradient error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        return image_gradient(pred, target)



@register("mse")
class MSE(Metric):
    """Computes weighted mean-squared error with variable-specific weights."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        var_names: Optional[List[str]] = None,
        var_weights: Optional[Dict[str, float]] = None
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        """
        Compute the weighted MSE loss.
        
        Args:
            pred: Predictions tensor of shape [B,C,H,W]
            target: Target tensor of shape [B,C,H,W]
            
        Returns:
            Loss tensor
        """
        return mse(
            pred,
            target,
            var_names,
            var_weights,
            self.aggregate_only
        )



@register("mae")
class MAE(Metric):
    """Computes L1 norm error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MAEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mae(pred, target, self.aggregate_only)



@register("lat_mse")
class LatWeightedMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        return mse(pred, target, self.aggregate_only, self.lat_weights)


@register("rmse")
class RMSE(Metric):
    """Computes standard root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, mask)
        return rmse(pred, target, self.aggregate_only)


@register("lat_rmse")
class LatWeightedRMSE(LatitudeWeightedMetric):
    """Computes latitude-weighted root mean-squared error."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return rmse(pred, target, self.aggregate_only, self.lat_weights, mask)
        return rmse(pred, target, self.aggregate_only, self.lat_weights)


@register("acc")
class ACC(ClimatologyBasedMetric):
    """
    Computes standard anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().cast_to_device(pred)
        if mask is not None:
            return acc(pred, target, self.climatology, self.aggregate_only, mask)
        return acc(pred, target, self.climatology, self.aggregate_only)


@register("lat_acc")
class LatWeightedACC(LatitudeWeightedMetric, ClimatologyBasedMetric):
    """
    Computes latitude-weighted anomaly correlation coefficient.
    """

    def __init__(self, *args, **kwargs):
        LatitudeWeightedMetric.__init__(self, *args, **kwargs)
        ClimatologyBasedMetric.__init__(self, *args, **kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
        mask=None,
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        LatitudeWeightedMetric.cast_to_device(self, pred)
        ClimatologyBasedMetric.cast_to_device(self, pred)
        if mask is not None:
            return acc(
                pred,
                target,
                self.climatology,
                self.aggregate_only,
                self.lat_weights,
                mask,
            )
        return acc(
            pred, target, self.climatology, self.aggregate_only, self.lat_weights
        )


@register("pearson")
class Pearson(Metric):
    """
    Computes the Pearson correlation coefficient, based on
    https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/10
    """

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            Pearson correlation coefficient, and the preceding elements are the
            channel-wise Pearson correlation coefficients.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return pearson(pred, target, self.aggregate_only)


@register("mean_bias")
class MeanBias(Metric):
    """Computes the standard mean bias."""

    def __call__(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor],
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate mean
            bias, and the preceding elements are the channel-wise mean bias.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        return mean_bias(pred, target, self.aggregate_only)
