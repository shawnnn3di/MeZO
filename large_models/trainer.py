# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np
import logging as py_logging

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers


_is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            # else:
            #     metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            #     logger.info(metrics)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
            
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
            # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
            optimizer = self.optimizer.optimizer
        else:
            optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
            
        print(self.optimizer)

        return self.optimizer
            
            
    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        mSGD_kwargs = {
            "momentum": args.momentum
        }
        
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_kwargs.update(mSGD_kwargs)
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

            

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        
        
        # 创建一个文件处理器，设置日志文件路径
        file_handler = py_logging.FileHandler(os.path.join(self.args.output_dir, 'log'))
        file_handler.setLevel(py_logging.INFO)
        # 将处理器添加到 logger
        logger.addHandler(file_handler)
        
        self.g = torch.Generator(device='cuda')
        
        self.update_once = False if args.zoo_name in ['mSGD_3', 'mSGD_2', 'mSGD_1', 'adam', 'anderson'] else True

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)
            
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9998)
        if args.zoo_name == 'mSGD_2':
            # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [5000, 10000, 15000], gamma=0.31622776601683793319988935444327)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9998)
            # pass
        if args.zoo_name == 'anderson':
            self.alpha_buffer = np.zeros((args.window_width, args.window_width + 1))
            self.proj_buffer = np.zeros((args.window_width, args.window_width + 1))

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        self.total_step = 0
        timer = 0
        random_seed = None
        # if args.seedprune > 0:
        #     assert args.seedprune < 1, 'prune prob should be in [0, 1)'
        #     self.explored = np.array([])
        #     self.sampled = np.array([])
        #     self.accumulated_projection = np.array([])
        # elif args.scope > 0:
        
        self.history_loss = []
        self.lowest = 9
        self.counter = 0
        
        if args.zoo_name == 'adam':
            self.adam = True
        else:
            self.adam = False
            
        if args.scope > 0:
            print('SCOPE: %d' % args.scope)
            self.weight = np.array([1 / args.scope] * args.scope)
            self.sampled = np.array([0] * args.scope)
            self.accumulated_projection = np.array([0.0] * args.scope)
        if args.window_width > 0:
            self.history_projection = []
            self.history_seed = []
            
        if self.adam:
            self.anderson_beta1 = np.array([(args.beta1) ** (args.window_width-i) for i in range(args.window_width)])
            self.anderson_beta2 = np.array([(args.beta2) ** (args.window_width-i) for i in range(args.window_width)])
        
        print(self.optimizer)
        
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            metrics['relative_time'] = timer
            logger.info(metrics)
            
            for step, inputs in enumerate(epoch_iterator):
                
                start = time.time()
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    
                # ismezo
                # if (args.seedprune > 0) & (args.sample_scheme == 'seedprune'):
                #     if self.total_step > 0:
                #         explore_flag = np.random.rand() > args.seedprune
                #     else:
                #         explore_flag = True
                #         random_seed = np.array([0])
                #     if explore_flag:
                #         random_seed = len(self.explored)
                #     else:
                #         random_seed = np.random.choice(self.explored, 1, p=self.weight)
                # elif args.scope > 0:
                if args.scope > 0:
                    if args.sample_scheme in ['kseedpro', 'is', 'is_v1']:
                        random_seed = np.random.choice(args.scope, 1, p=self.weight)
                    elif args.sample_scheme == 'default':
                        random_seed = np.random.choice(args.scope, 1)
                    

                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    tr_loss_step = self.zo_step(model, inputs, random_seed)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self.history_loss.append(tr_loss_step.detach().cpu().numpy())
                
                # ismezo, update weight
                # todo
                # kseedpro
                self.total_step += 1
                
                if args.trainer == 'zo':
                    if args.sample_scheme == 'kseedpro':
                        print('kseedpro')
                        self.sampled[random_seed] += 1
                        self.accumulated_projection[random_seed] += np.abs(self.projected_grad)
                        self.explored_projection_mean = np.sum(self.accumulated_projection[self.sampled > 0]) / self.total_step
                        self.weight[self.sampled > 0] = self.accumulated_projection[self.sampled > 0] / self.sampled[self.sampled > 0]
                        self.weight[self.sampled == 0] = self.explored_projection_mean
                        self.weight /= np.sum(self.weight)
                        
                        self.weight *= np.sum(self.accumulated_projection) / self.total_step
                        self.weight = np.exp(self.weight)
                        self.weight /= np.sum(self.weight)
                    if args.sample_scheme == 'is':
                        print('is')
                        self.accumulated_projection[random_seed] += np.abs(self.projected_grad)
                        self.projected_grad *= 1 / args.scope / self.weight[random_seed][0]
                        
                        # self.accumulated_projection[random_seed] += np.abs(self.projected_grad)
                        self.sampled[random_seed] += 1
                        self.explored_projection_mean = np.sum(self.accumulated_projection[self.sampled > 0]) / self.total_step
                        self.weight[self.sampled > 0] = self.accumulated_projection[self.sampled > 0] / self.sampled[self.sampled > 0]
                        self.weight[self.sampled == 0] = self.explored_projection_mean
                        self.weight += 1e-8
                        self.weight /= np.sum(self.weight)
                    if args.sample_scheme == 'is_v1':
                        print('is_v1')
                        self.accumulated_projection[random_seed] = 0.8 * self.accumulated_projection[random_seed] + 0.2 * np.abs(self.projected_grad)
                        self.projected_grad *= 1 / args.scope / self.weight[random_seed][0]
                        
                        # self.accumulated_projection[random_seed] += np.abs(self.projected_grad)
                        self.sampled[random_seed] += 1
                        self.explored_projection_mean = np.sum(self.accumulated_projection[self.sampled > 0]) / self.total_step
                        self.weight[self.sampled > 0] = self.accumulated_projection[self.sampled > 0] / self.sampled[self.sampled > 0]
                        self.weight[self.sampled == 0] = self.explored_projection_mean
                        self.weight += 1e-8
                        self.weight /= np.sum(self.weight)
                    # if (args.seedprune > 0) & (args.sample_scheme == 'seedprune'):
                    #     print('seedprune')
                    #     if explore_flag:
                    #         self.explored = np.append(self.explored, random_seed).astype(np.int64)
                    #         self.accumulated_projection = np.append(self.accumulated_projection, np.abs(self.projected_grad))
                    #         self.weight = self.accumulated_projection + 1e-8
                    #         self.weight /= np.sum(self.weight)
                        # else:
                            # self.accumulated_projection[random_seed] = args.refresh * np.abs(self.projected_grad) + (1 - args.refresh) * self.accumulated_projection[random_seed]
                            # self.projected_grad *= 1 / len(self.weight) / self.weight[random_seed][0]
                            # self.weight += 1e-8
                            # self.weight /= np.sum(self.weight)
                    if args.window_width > 0:
                        # self.history_projection.append(np.abs(self.projected_grad))
                        self.history_projection.append(self.projected_grad)
                        self.history_seed.append(self.zo_random_seed)

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        self.zo_update(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                
                timer += time.time() - start
                logger.info('mylog, %d, %.4f, %.4f, %.4f, %d, %.4f' % (self.total_step, tr_loss_step, tr_loss, self.lowest, self.counter, timer))
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
                
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## MeZO ##############


    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.g.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=self.g)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
        # print(self.g.get_state())


    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)


    def zo_step(self, model, inputs, random_seed=None):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = random_seed if random_seed is not None else np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        
        self.zo_perturb_parameters(scaling_factor=1)
        loss0 = self.zo_forward(model, inputs)
        
        
        df = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        pddf = ((loss2 - 2 * loss0 + loss1) / (self.args.zo_eps ** 2)).item() + 1e-8
        if np.abs(pddf) < 1e-5:
            pddf = 1e-5
        ddf = np.clip(pddf, -1e5, 1e5)

        # self.projected_grad = np.clip(df / ddf, -1e5, 1e-5)
        self.projected_grad = df
        
        print('%.4f, %.4f, %.4f' % (df, ddf, self.projected_grad))

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        # self.zo_perturb_parameters(scaling_factor=1)
        
        return loss0
        
        # self.zo_perturb_parameters(scaling_factor=-2)
        # loss_l2 = self.zo_forward(model, inputs)
        # self.zo_perturb_parameters(scaling_factor=1)
        # loss_l1 = self.zo_forward(model, inputs)
        # self.zo_perturb_parameters(scaling_factor=2)
        # loss_r1 = self.zo_forward(model, inputs)
        # self.zo_perturb_parameters(scaling_factor=1)
        # loss_r2 = self.zo_forward(model, inputs)
        # self.zo_perturb_parameters(scaling_factor=-2)
        # loss_c = self.zo_forward(model, inputs)
        
        # df = ((loss_l2 - 8 * loss_l1 + 8 * loss_r1 - loss_r2) / (self.args.zo_eps * 12)).item()
        # ddf = float(np.clip( - ((loss_l2 - 16 * loss_l1 + 30 * loss_c - 16 * loss_r1 + loss_r2) / (self.args.zo_eps ** 2 * 12)).item(), -1e5, 1e5))
        # ddf = np.clip(np.abs(ddf), 0.001, 1e5) ** 0.25 #* ((ddf > 0) - 0.5) * 2
        # self.projected_grad = np.clip(df / ddf, -1e5, 1e5)
        # # self.projected_grad = df 
        
        # print(np.clip( - ((loss_l2 - 16 * loss_l1 + 30 * loss_c - 16 * loss_r1 + loss_r2) / (self.args.zo_eps ** 2 * 12)).item(), -1e5, 1e5))
        
        # print('%.4f, %.4f, %.4f' % (df, ddf, self.projected_grad))
        
        # assert self.args.gradient_accumulation_steps == 1
        
        # return loss_c


    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        k = args.window_width
        m = min(k, len(self.history_seed))

        # if self.total_step == args.stop_momentum:
        #     self.update_once = True
        if args.zoo_name in ['mSGD_1', 'mSGD_3', 'adam',  'mSGD_2', 'anderson']:
            n = 200
            if self.total_step > n:
                self.record_loss = np.mean(np.array(self.history_loss[-n:]))
                if self.record_loss < self.lowest:
                    self.lowest = self.record_loss
                    self.counter = 0
                else:
                    self.counter += 1
                if self.counter > 40000:
                    if self.update_once == False:
                        logger.info('abandon momentum at step %d' % self.total_step)
                    self.update_once = True

        # Reset the random seed for sampling zs
        
        if not self.update_once:
            if not args.zoo_name == 'anderson':
                
                # anderson_projection = np.clip(self.history_projection[-m:], 1e-8, 100000)
                if args.zoo_name == 'mSGD_1':
                    # self.anderson_weight = np.array([(args.mu * np.max([0, (args.mu - self.total_step / args.max_steps)])) ** (args.window_width-i-1) for i in range(args.window_width)])
                    
                    self.anderson_weight = np.abs(np.array(self.history_projection[-m:]))
                    self.anderson_weight = self.anderson_weight / np.sum(self.anderson_weight) * m * np.array([args.mu ** (args.window_width-i-1) for i in range(args.window_width)])[-m:]
                    anderson_projection = np.clip(self.history_projection[-m:], -1e8, 1e8)
                    anderson_seed = self.history_seed[-m:]
                    
                elif args.zoo_name == 'mSGD_2':
                    # anderson_projection = np.clip(self.history_projection[-m:], -1e5, 1e5)
                    # anderson_seed = self.history_seed[-m:]
                    # self.anderson_weight = 1 / np.clip(np.abs(anderson_projection), 1e-5, 1e5)
                    # self.anderson_weight /= np.sum(self.anderson_weight)
                    # sample_weight = np.abs(anderson_projection)
                    # sample_weight /= np.sum(sample_weight)
                    # idx = np.random.choice(m, m, p=sample_weight)
                    # anderson_seed = np.array(anderson_seed)[idx]
                    # self.anderson_weight = self.anderson_weight[idx]
                    
                    self.anderson_weight = 1 / np.array(self.history_loss[-m:])
                    self.anderson_weight = self.anderson_weight / np.sum(self.anderson_weight) * m * np.array([args.mu ** (args.window_width-i-1) for i in range(args.window_width)])[-m:]
                    anderson_projection = np.clip(self.history_projection[-m:], -1e8, 1e8)
                    anderson_seed = self.history_seed[-m:]
                    
                elif args.zoo_name in ['mSGD_3', 'adam']:

                    self.anderson_weight = np.array([args.mu ** (args.window_width-i-1) for i in range(args.window_width)])
                    anderson_projection = np.clip(self.history_projection[-m:], -1e8, 1e8)
                    anderson_seed = self.history_seed[-m:]
                    
                if not self.adam:
                    for i in range(m):
                        torch.manual_seed(anderson_seed[i])
                        self.g.manual_seed(anderson_seed[i])
                        for name, param in self.named_parameters_to_optim:
                            # Resample z
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            param.data = param.data - self._get_learning_rate() * anderson_projection[i] * self.anderson_weight[i] * z
                            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                                param.data = param.data - self._get_learning_rate() * (args.weight_decay * param.data)
                    if args.nesterov:
                        torch.manual_seed(anderson_seed[-1])
                        self.g.manual_seed(anderson_seed[-1])
                        for name, param in self.named_parameters_to_optim:
                            # Resample z
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            param.data = param.data - self._get_learning_rate() * (anderson_projection[-1] * self.anderson_weight[-1] * z)
                else:
                    
                    state_list = []
                    for i in range(m):
                        torch.manual_seed(anderson_seed[i])
                        self.g.manual_seed(anderson_seed[i])
                        state_list.append(self.g.get_state())
                        
                    for name, param in self.named_parameters_to_optim:
                        for i in range(m):
                            self.g.set_state(state_list[i])
                            _m = torch.zeros(size=param.data.size(), device=param.data.device, dtype=param.data.dtype, requires_grad=False)
                            _v = torch.zeros(size=param.data.size(), device=param.data.device, dtype=param.data.dtype, requires_grad=False)
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=self.g)
                            state_list[i] = self.g.get_state()
                            
                            if i != m-1:
                                if m != 1:
                                    _m += self.anderson_beta1[-(m-1):][i] * (anderson_projection[i] * z)
                                    _v += self.anderson_beta2[-(m-1):][i] * ((anderson_projection[i] * z) ** 2)
                                else:
                                    pass
                            else:
                                # _m += (1 - args.beta1) * z * anderson_projection[-1]
                                # _v += (1 - args.beta2) * (z ** 2) * anderson_projection[-1]
                                _m += (1 - args.beta1) * z * anderson_projection[-1]
                                _v += (1 - args.beta2) * (z * anderson_projection[-1]) ** 2
                                # _m = z * anderson_projection[-1]
                                # _v = (z * anderson_projection[-1]) ** 2
                            
                            # _v = (1 - args.beta2) * (z * anderson_projection[-1] - _m) ** 2
                            
                            # _m /= (1 - args.beta1 ** self.total_step)
                            # _v /= (1 - args.beta2 ** self.total_step)
                            _v = torch.clip(_v, 0.000976562, 66503)
                            # _v /= _v.mean()
                            # print(_v.mean())
                            
                            # _m += z * anderson_projection[i]
                            
                            # param.data = param.data - self._get_learning_rate() * torch.clip(_m / (_v ** 0.5 + 1e-7), -1e4, 1e4)
                            # param.data = param.data - torch.clip(self._get_learning_rate() / (torch.sqrt(_v)), 0.000976562, 66503) * _m
                            
                            # v1
                            # grad = _m / (torch.sqrt(_v) * ((1 - args.beta1 ** self.total_step) / (1 - args.beta2 ** self.total_step)) + 1e-5)
                            # v2
                            grad = _m / (_v * ((1 - args.beta1 ** self.total_step) / (1 - args.beta2 ** self.total_step)) + 1e-5)
                            
                            # gradnorm = torch.mean(grad)
                            # if gradnorm > 0.01:
                            #     grad *= 0.01 / gradnorm
                            param.data = param.data - self._get_learning_rate() * (grad / (torch.norm(grad) + 1e-5))
                            # param.data = param.data - self._get_learning_rate() * torch.clip(_m / ((_v ** 0.5).max() + 1e-8), -1e10, 1e10)
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data = param.data - self._get_learning_rate() * (args.weight_decay * param.data)
                        
                    a = 1
                
            else:
                torch.manual_seed(self.zo_random_seed)
                self.g.manual_seed(self.zo_random_seed)
                for name, param in self.named_parameters_to_optim:
                    # Resample z
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype, generator=self.g)
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                    else:
                        param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
                        
                if self.total_step % args.window_width == 0:
                    mix_weight = 1 / np.array(self.history_loss[-args.window_width:])
                    mix_weight /= np.sum(mix_weight)
                    for i in range(args.window_width):
                        self.zo_random_seed = self.history_seed[-i]
                        self.projected_grad = self.history_projection[-i]
                        torch.manual_seed(self.zo_random_seed)
                        self.g.manual_seed(self.zo_random_seed)
                        for name, param in self.named_parameters_to_optim:
                            # Resample z
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                                param.data = param.data + self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                            else:
                                param.data = param.data + self._get_learning_rate() * (self.projected_grad * z)
                    
                    for i in range(args.window_width):
                        self.zo_random_seed = self.history_seed[-i]
                        self.projected_grad = self.history_projection[-i]
                        w = (1 - np.sum(mix_weight[-i:]))
                        torch.manual_seed(self.zo_random_seed)
                        self.g.manual_seed(self.zo_random_seed)
                        for name, param in self.named_parameters_to_optim:
                            # Resample z
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data) * w
                            else:
                                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z) * w
                
                
                inner_step = self.total_step % args.window_width
                if (inner_step - 1) == 0:
                    self.alpha_buffer = np.zeros((args.window_width, args.window_width + 1))
                    self.proj_buffer = np.zeros((args.window_width, args.window_width + 1))
                    self.alpha_buffer[0][0] = 1
                    self.alpha_buffer[0][1] = 1
                    self.proj_buffer[0][0] = 1
                    self.proj_buffer[0][1] = self.projected_grad
                    
                    torch.manual_seed(self.zo_random_seed)
                    self.g.manual_seed(self.zo_ranmdom_seed)
                    for name, param in self.named_parameters_to_optim:
                        # Resample z
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                        else:
                            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
                else:
                    if inner_step == 0:
                        inner_step = args.window_width
                    # mix_weight = np.tan(np.clip(self.history_loss[-inner_step:], 0, 3.1415926/2))
                    mix_weight = np.array(self.history_loss[-inner_step:])
                    mix_weight = 1 / mix_weight
                    mix_weight /= np.sum(mix_weight)
                    self.alpha_buffer[inner_step - 1][0] = 1
                    self.proj_buffer[inner_step - 1][0] = 1
                    self.alpha_buffer[inner_step - 1] = np.matmul(mix_weight, self.alpha_buffer[:inner_step])
                    self.alpha_buffer[inner_step - 1][inner_step] = 1
                    self.proj_buffer[inner_step - 1] = np.matmul(mix_weight, self.proj_buffer[:inner_step])
                    self.proj_buffer[inner_step - 1][inner_step] = self.projected_grad
                    
                    for i in range(inner_step):
                        torch.manual_seed(self.history_seed[-i])
                        self.g.manual_seed(self.history_seed[-i])
                        projected_grad = self.proj_buffer[inner_step - 1][-i] - self.proj_buffer[inner_step - 2][-i]
                        for name, param in self.named_parameters_to_optim:
                            # Resample z
                            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                                param.data = param.data - self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
                            else:
                                param.data = param.data - self._get_learning_rate() * (projected_grad * z)
                
                
                
                # inner_step = self.total_step % args.window_width
                # if (inner_step - 1) != 0:
                
                
                # torch.manual_seed(self.zo_random_seed)
                # for name, param in self.named_parameters_to_optim:
                #     # Resample z
                #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                #     if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                #         param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                #     else:
                #         param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
                        
                # torch.manual_seed(torch.randint(100000000000000, (1, )))
                # for name, param in self.named_parameters_to_optim:
                #     # Resample z
                #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype) * 0.03
                #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
                        

            # else:
            #     self.update_once = True
            
            # logger.info(f'{anderson_projection}')


                            # print(name)
                            # print(torch.isnan(param.data).sum())
        else:
            torch.manual_seed(self.zo_random_seed)
            self.g.manual_seed(self.zo_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        self.lr_scheduler.step()


    ############## Misc overload functions ##############


    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
            
        # ismezo
        if self.args.scope > 0:
            import pickle as pk
            pk.dump([self.sampled, self.weight, self.accumulated_projection], open(os.path.join(output_dir, 'mymetric.pk'), 'wb'))
