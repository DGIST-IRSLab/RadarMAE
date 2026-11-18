'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import numpy as np
import open3d as o3d
import os
import psutil
from tqdm import tqdm
import shutil
from torch.utils.data import Subset

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file
from utils.pretrain_inspect import apply_to_finetune_cfg

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.kitti_eval.eval_revised import get_official_eval_result_revised

from utils.util_optim import clip_grad_norm_


def _torch_save_atomic(obj, path):
    path = os.fspath(path)
    tmp_path = f'{path}.tmp.{os.getpid()}'
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


class PipelineDetection_v1_0():
    def __init__(self, path_cfg=None, mode='train'):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        self.update_cfg_regarding_mode()

        # PRETRAIN_PATH가 지정돼 있고 AUTO_OVERRIDE_FROM_PRETRAIN != False면
        # ckpt에서 tokenizer/patch/stem/pe/input_norm 등을 자동 추출해 cfg.MAE 덮어씀.
        try:
            mae_cfg = self.cfg.MODEL.BACKBONE.MAE
            auto = bool(mae_cfg.get('AUTO_OVERRIDE_FROM_PRETRAIN', True))
        except Exception:
            auto = False
        if auto:
            apply_to_finetune_cfg(self.cfg, ckpt_path=None, verbose=True)
        self.is_logging = False
        self.log_iter_start = None

        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)
        
        print('* K-Radar dataset is being loaded.')
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        # Load validation set if split contains val.txt (3rd element)
        data_splits = list(self.cfg.DATASET.path_data.split)
        has_val_split = len(data_splits) > 2 and os.path.abspath(data_splits[2]) != os.path.abspath(data_splits[1])
        self.dataset_val = build_dataset(self, split='val') if has_val_split else self.dataset_test
        print('* The dataset is loaded.')
        if mode == 'train': # for setting scheduler
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated

        self.network = build_network(self).cuda()
        two_phase_cfg = self.cfg.OPTIMIZER.get('TWO_PHASE', None)
        if two_phase_cfg is not None and two_phase_cfg.get('ENABLED', False):
            backbone_lr_scale = two_phase_cfg.get('PHASE1_BACKBONE_LR_SCALE', 0.1)
        else:
            backbone_lr_scale = None
        self.optimizer = build_optimizer(self, self.network, backbone_lr_scale=backbone_lr_scale)
        self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0
        # WandB 기본값은 로깅 설정 이전에 잡아두고, set_logging에서 활성화 여부를 결정한다.
        self.wandb_run = None
        self.use_wandb = False
        self.wandb = None
        self.wandb_global_step = 0
        self.wandb_iter_offset = 0
        self.best_metric = float('-inf')
        self.best_metric_name = None
        self.bev_ap_plot_history = {'subset': [], 'total': []}
        self.bev_ap_plot_warned = False
        self.bev_ap_plot_path = os.environ.get('FINETUNE_BEV_AP_PLOT_PATH', '').strip()
        self.bev_ap_plot_iou = self._get_float_env('FINETUNE_BEV_AP_PLOT_IOU', 0.3)
        self.bev_ap_plot_conf = self._get_float_env('FINETUNE_BEV_AP_PLOT_CONF', None)

        # Logging
        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False
        
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True \
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        # Vis
        self.set_vis()

        # self.show_pline_description()
        
        # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
        # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
        self.is_validation_updated = self.cfg.get('is_validation_updated', False)

        ### Distil ###
        cfg_distil = self.cfg.get('DISTIL', None)
        if cfg_distil is not None:
            self.distil = True
            self.infer_head_of_distil_model = cfg_distil.get('INFER_HEAD', False)
            import yaml
            from easydict import EasyDict
            with open(cfg_distil.CFG, 'r') as f:
                new_config = yaml.safe_load(f)
            from models.skeletons import build_skeleton
            distil_model = build_skeleton(EasyDict(new_config))
            
            if not self.infer_head_of_distil_model:
                if hasattr(distil_model, 'head'):
                    import torch.nn as nn
                    distil_model.head = nn.Identity()
            distil_model.load_state_dict(torch.load(cfg_distil.PTH), strict=False)
            self.distil_model = distil_model.cuda().eval()
            # inactivate self.head in distilation model
            print('* The model for distilation is loaded.')
        else:
            self.distil = False
        ### Distil ###

    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        if self.cfg_dataset_ver2:
            pass # TODO
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
            self.dict_cls_id_to_name = dict()
            for k, v in self.dict_cls_name_to_id.items():
                if v != -1:
                    self.dict_cls_id_to_name[v] = k
            self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
            self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB
    
    def show_pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        cfg_name = os.path.splitext(os.path.basename(path_cfg))[0]
        self.wandb_default_run_name = f'{str_local_time}_{cfg_name}'
        str_exp_base = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        path_logging_root = self.cfg.GENERAL.LOGGING.PATH_LOGGING

        # Avoid run failure when multiple jobs start in the same second and collide on exp folder name.
        self.exp_name = None
        self.path_log = None
        for idx_try in range(1000):
            str_exp = str_exp_base if idx_try == 0 else f'{str_exp_base}_{idx_try:03d}'
            path_log = os.path.join(path_logging_root, str_exp)
            try:
                os.makedirs(path_log, exist_ok=False)
                self.exp_name = str_exp
                self.path_log = path_log
                break
            except FileExistsError:
                continue

        if self.path_log is None:
            raise RuntimeError(f'* Exception error (Pipeline): cannot allocate unique log directory under {path_logging_root}')

        if is_print_where:
            print(f'* Start logging in {self.exp_name}')

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None
        self._init_wandb()

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        if not self.bev_ap_plot_path:
            safe_iou = str(self.bev_ap_plot_iou).replace('.', 'p')
            self.bev_ap_plot_path = os.path.join(
                self.path_log,
                f'{cfg_name}_bev_ap_iou{safe_iou}.png',
            )

        # code backup (TBD)

    def _init_wandb(self):
        wandb_cfg = getattr(self.cfg.GENERAL.LOGGING, 'WANDB', None)
        enable_flag = False
        if wandb_cfg is not None:
            enable_flag = wandb_cfg.get('ENABLE', False) or wandb_cfg.get('ENABLED', False)
        if not enable_flag:
            self.use_wandb = False
            return
        try:
            import wandb
        except ImportError:
            print('* Warning: wandb is not installed. Disable WANDB logging.')
            self.use_wandb = False
            return
        # If a local folder named "wandb" exists in the working directory, it can shadow
        # the real W&B package and break with AttributeError (e.g., no wandb.init).
        if not hasattr(wandb, 'init'):
            try:
                src = getattr(wandb, '__file__', None) or str(getattr(wandb, '__path__', ''))
            except Exception:
                src = None
            print(f"* Warning: imported 'wandb' has no attribute 'init' (shadowed module?). src={src}. Disable WANDB logging.")
            self.use_wandb = False
            return

        # optional env/dir/mode handling (align to K-Radar behavior)
        wandb_dir = wandb_cfg.get('DIR')
        if wandb_dir:
            os.environ['WANDB_DIR'] = wandb_dir
        wandb_mode = wandb_cfg.get('MODE')
        if wandb_mode:
            os.environ['WANDB_MODE'] = wandb_mode

        run_name = wandb_cfg.get('RUN_NAME') or getattr(self, 'wandb_default_run_name', None) or getattr(self, 'exp_name', None)
        run_kwargs = {
            'project': wandb_cfg.get('PROJECT', 'kradar-detection'),
            'entity': wandb_cfg.get('ENTITY', None),
            'notes': wandb_cfg.get('NOTES', None),
            'tags': wandb_cfg.get('TAGS') if wandb_cfg.get('TAGS') else None,
            'name': run_name,
            'reinit': True,
            'config': self._config_to_dict(self.cfg)
        }
        resume_cfg = wandb_cfg.get('RESUME', False)
        if resume_cfg:
            run_kwargs['resume'] = resume_cfg
        self.wandb_run = wandb.init(**run_kwargs)
        self.use_wandb = self.wandb_run is not None
        self.wandb = wandb
        self.wandb_iter_offset = 0 if self.log_iter_start is None else self.log_iter_start

    def _config_to_dict(self, node):
        # EasyDict를 먼저 dict로 변환해서 origin과 동일한 wandb config 포맷을 사용
        try:
            from easydict import EasyDict
            if isinstance(node, EasyDict):
                node = dict(node)
        except Exception:
            pass
        if isinstance(node, dict):
            return {k: self._config_to_dict(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [self._config_to_dict(v) for v in node]
        return node

    def _finish_wandb(self):
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    @staticmethod
    def _get_float_env(name, default):
        value = os.environ.get(name, None)
        if value is None or str(value).strip() == '':
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_wandb_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return float(value.detach().mean().item())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)) and value:
            try:
                return float(value[0])
            except Exception:
                pass
        try:
            return float(value)
        except Exception:
            return value

    def _wandb_log(self, log_dict, step=None, commit=True):
        if not self.use_wandb or self.wandb_run is None:
            return
        if step is None:
            step = self.wandb_global_step
        else:
            self.wandb_global_step = step
        self.wandb.log(log_dict, step=step, commit=commit)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict_dir = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP

        # Prefer util_last.pt (current save format), but tolerate interrupted
        # torch.save files by falling back to numbered utils or util_best.pt.
        list_candidates = []
        path_util_last = os.path.join(path_state_dict_dir, 'util_last.pt')
        path_util_best = os.path.join(path_state_dict_dir, 'util_best.pt')
        if epoch is not None:
            list_candidates.append(os.path.join(path_state_dict_dir, f'util_{epoch}.pt'))
        else:
            list_epochs = sorted([
                int(x.split('.')[0].split('_')[1])
                for x in os.listdir(path_state_dict_dir)
                if x.startswith('util_') and x != 'util_last.pt' and x.split('.')[0].split('_')[1].isdigit()
            ])
            list_candidates.append(path_util_last)
            for ep in reversed(list_epochs):
                list_candidates.append(os.path.join(path_state_dict_dir, f'util_{ep}.pt'))
            list_candidates.append(path_util_best)

        list_candidates = [p for p in dict.fromkeys(list_candidates) if os.path.exists(p)]
        if not list_candidates:
            raise FileNotFoundError(f'No checkpoint to resume from in {path_state_dict_dir}')

        state_dict = None
        path_state_dict = None
        last_error = None
        for candidate in list_candidates:
            print('* Try resume, path_state_dict =  ', candidate)
            try:
                state_dict = torch.load(candidate)
                path_state_dict = candidate
                break
            except Exception as e:
                last_error = e
                print(f'* Warning: skip invalid resume checkpoint: {candidate} ({type(e).__name__}: {e})')
        if state_dict is None:
            raise RuntimeError(f'No loadable checkpoint to resume from in {path_state_dict_dir}') from last_error

        print('* Start resume, path_state_dict =  ', path_state_dict)
        ckpt_epoch = int(state_dict.get('epoch', epoch if epoch is not None else 0))

        try:
            self.epoch_start = ckpt_epoch + 1
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            if 'best_metric' in state_dict:
                self.best_metric = state_dict['best_metric']
                self.best_metric_name = state_dict.get('best_metric_name', None)
            print(f'* Network & Optimizer are loaded / Resume epoch is {ckpt_epoch} / Start from {self.epoch_start} ...')
        except:
            print('* Exception error (Pipeline): check resume network')
            exit()

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                src_dir = os.path.join(path_exp, copy_dir)
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    @staticmethod
    def _format_gib(num_bytes):
        return f'{float(num_bytes) / (1024 ** 3):.1f}G'

    @staticmethod
    def _get_job_memory_usage():
        try:
            process = psutil.Process(os.getpid())
            used_bytes = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    used_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            return None

        return used_bytes

    @staticmethod
    def _get_job_memory_limit_bytes():
        limit_gb = os.environ.get('KRADAR_JOB_RAM_LIMIT_GB', None)
        if limit_gb is not None:
            try:
                return int(float(limit_gb) * (1024 ** 3))
            except Exception:
                pass

        limit_mb = os.environ.get('SLURM_MEM_PER_NODE', None)
        if limit_mb is not None:
            try:
                return int(float(limit_mb) * (1024 ** 2))
            except Exception:
                pass

        return None

    def _get_training_memory_postfix(self):
        postfix = {}

        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                used_bytes = max(total_bytes - free_bytes, 0)
                postfix['vram'] = f"{self._format_gib(used_bytes)}/{self._format_gib(total_bytes)}"
                peak_bytes = torch.cuda.max_memory_allocated(device=device)
                postfix['vram_peak'] = self._format_gib(peak_bytes)
            except Exception:
                pass

        ram_used_bytes = self._get_job_memory_usage()
        if ram_used_bytes is not None:
            ram_limit_bytes = self._get_job_memory_limit_bytes()
            if ram_limit_bytes is not None:
                postfix['ram'] = f"{self._format_gib(ram_used_bytes)}/{self._format_gib(ram_limit_bytes)}"
            else:
                postfix['ram'] = self._format_gib(ram_used_bytes)

        return postfix

    @staticmethod
    def _mean_bev_ap_from_metrics(list_metrics):
        bev_values = []
        for dict_metrics in list_metrics:
            for bev_ap in dict_metrics.get('bev', []):
                try:
                    bev_values.append(float(bev_ap))
                except (TypeError, ValueError):
                    continue

        if len(bev_values) == 0:
            return None
        return float(np.mean(bev_values))

    @staticmethod
    def _mean_bev_ap_at_iou_from_metrics(list_metrics, target_iou=0.3):
        bev_values = []
        for dict_metrics in list_metrics:
            ious = dict_metrics.get('iou', [])
            bevs = dict_metrics.get('bev', [])
            if not ious or not bevs:
                continue
            try:
                iou_values = [float(v) for v in ious]
            except (TypeError, ValueError):
                continue
            idx_iou = int(np.argmin([abs(v - float(target_iou)) for v in iou_values]))
            if idx_iou >= len(bevs):
                continue
            try:
                bev_values.append(float(bevs[idx_iou]))
            except (TypeError, ValueError):
                continue

        if len(bev_values) == 0:
            return None
        return float(np.mean(bev_values))

    def _select_bev_ap_plot_conf(self, list_conf_thr, metrics_by_conf):
        if not list_conf_thr:
            return None
        if self.bev_ap_plot_conf is not None:
            try:
                return min(metrics_by_conf.keys(), key=lambda x: abs(float(x) - self.bev_ap_plot_conf))
            except Exception:
                return self.bev_ap_plot_conf
        return list_conf_thr[0]

    def _record_bev_ap_plot_point(self, epoch, split_name, conf_thr, bev_ap):
        if epoch is None or bev_ap is None or not self.bev_ap_plot_path:
            return

        series = self.bev_ap_plot_history.setdefault(split_name, [])
        epoch_int = int(epoch)
        series[:] = [item for item in series if item[0] != epoch_int]
        series.append((epoch_int, float(bev_ap), float(conf_thr)))
        series.sort(key=lambda item: item[0])
        self._write_bev_ap_plot()

    def _write_bev_ap_plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt

            path_plot = self.bev_ap_plot_path
            path_plot_dir = os.path.dirname(path_plot)
            if path_plot_dir:
                os.makedirs(path_plot_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            plotted = False
            for split_name, label, marker in (
                ('subset', 'subset val', 'o'),
                ('total', 'total val', 's'),
            ):
                points = self.bev_ap_plot_history.get(split_name, [])
                if not points:
                    continue
                epochs = [p[0] + 1 for p in points]
                values = [p[1] for p in points]
                confs = sorted({p[2] for p in points})
                conf_label = ','.join(f'{c:g}' for c in confs)
                ax.plot(
                    epochs,
                    values,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4.5,
                    label=f'{label} (conf {conf_label})',
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                return

            ax.set_title(f'BEV AP @ IoU {self.bev_ap_plot_iou:g}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('BEV AP (%)')
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.45)
            ax.legend(loc='best')
            ax.set_ylim(bottom=0)
            fig.tight_layout()

            tmp_path = f'{path_plot}.tmp'
            fig.savefig(tmp_path, dpi=150, format='png')
            plt.close(fig)
            os.replace(tmp_path, path_plot)
            print(f'* BEV AP plot updated: {path_plot}')
        except Exception as e:
            if not self.bev_ap_plot_warned:
                print(f'* Warning: failed to update BEV AP plot: {repr(e)}')
                self.bev_ap_plot_warned = True

    def _transition_to_phase2(self):
        """Rebuild backbone with dense stride and create new optimizer/scheduler for Phase 2."""
        two_phase_cfg = self.cfg.OPTIMIZER.TWO_PHASE
        new_stride = two_phase_cfg.PHASE2_STRIDE
        phase2_lr = two_phase_cfg.get('PHASE2_LR', self.cfg.OPTIMIZER.LR)
        phase2_min_lr = two_phase_cfg.get('PHASE2_MIN_LR', self.cfg.OPTIMIZER.MIN_LR)

        print('=' * 60)
        print(f'* 2-Phase Transition: stride {list(self.network.backbone.stripe_patch_stride)} -> {new_stride}')
        print('=' * 60)

        # Save current model weights
        old_state = self.network.state_dict()

        # Update config for Phase 2 stride
        self.cfg.MODEL.BACKBONE.MAE.STRIPE_PATCH_STRIDE = new_stride
        # Disable pretrain loading (we already have detection-adapted weights)
        saved_pretrain_path = self.cfg.MODEL.BACKBONE.MAE.PRETRAIN_PATH
        self.cfg.MODEL.BACKBONE.MAE.PRETRAIN_PATH = ''

        # Rebuild backbone with new stride
        from models import backbone_2d
        new_backbone = backbone_2d.__all__[self.cfg.MODEL.BACKBONE.NAME](self.cfg).cuda()

        # Restore pretrain path for config consistency
        self.cfg.MODEL.BACKBONE.MAE.PRETRAIN_PATH = saved_pretrain_path

        # Transfer compatible weights from Phase 1
        new_state = new_backbone.state_dict()
        transferred = 0
        for k, v in old_state.items():
            if not k.startswith('backbone.'):
                continue
            bb_key = k[len('backbone.'):]
            if bb_key in new_state and v.shape == new_state[bb_key].shape:
                new_state[bb_key] = v
                transferred += 1
        new_backbone.load_state_dict(new_state)
        print(f'* Phase 2 backbone: transferred {transferred} weight tensors from Phase 1')

        # Replace backbone in network
        self.network.backbone = new_backbone

        # Rebuild list_modules to reflect the new backbone
        self.network.list_modules = []
        for name in self.network.list_module_names:
            if hasattr(self.network, name):
                self.network.list_modules.append(getattr(self.network, name))

        # Create new optimizer with uniform LR for Phase 2
        self.cfg.OPTIMIZER.LR = phase2_lr
        self.cfg.OPTIMIZER.MIN_LR = phase2_min_lr
        self.optimizer = build_optimizer(self, self.network, backbone_lr_scale=None)
        self.scheduler = build_scheduler(self, self.optimizer)
        print(f'* Phase 2 optimizer: LR={phase2_lr}, MIN_LR={phase2_min_lr}')
        print('=' * 60)

    def train_network(self, is_shuffle=True):
        self.network.train()
        data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
            batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
            collate_fn = self.dataset_train.collate_fn,
            num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        # 2-Phase config
        two_phase_cfg = self.cfg.OPTIMIZER.get('TWO_PHASE', None)
        two_phase_enabled = two_phase_cfg is not None and two_phase_cfg.get('ENABLED', False)
        phase1_epochs = two_phase_cfg.PHASE1_EPOCHS if two_phase_enabled else epoch_end
        phase_transitioned = False

        if self.optim_fastai:
            accumulated_iter = 0
            cfg_optim = self.cfg.OPTIMIZER
            use_amp = cfg_optim.get('USE_AMP', False)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=cfg_optim.get('LOSS_SCALE_FP16', 2.0**16))

        for epoch in range(epoch_start, epoch_end):
            # 2-Phase transition: switch stride at the start of Phase 2
            if two_phase_enabled and not phase_transitioned and epoch >= phase1_epochs:
                self._transition_to_phase2()
                phase_transitioned = True

            phase_str = ''
            if two_phase_enabled:
                phase_str = ' [Phase 1: stride match]' if epoch < phase1_epochs else ' [Phase 2: dense stride]'
            print(f'* Training epoch = {epoch}/{epoch_end-1}{phase_str}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')

            self.network.train()
            self.network.training = True
            avg_loss = []
            train_bar = tqdm(data_loader_train)
            for idx_iter, dict_datum in enumerate(train_bar):
                if self.optim_fastai:
                    self.scheduler.step(accumulated_iter, epoch)
                
                if self.distil:
                    with torch.no_grad():
                        dict_datum = self.distil_model(dict_datum)
                        dict_datum['ldr_bev_feat'] = dict_datum['spatial_features_2d']
                
                # try:
                dict_net = self.network(dict_datum)
                # except:
                #     print('* error: ', dict_datum['meta'])
                
                if self.get_loss_from == 'head':
                    loss = self.network.head.loss(dict_net)
                elif self.get_loss_from == 'detector':
                    loss = self.network.loss(dict_net)
                
                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if self.optim_fastai:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.network.parameters(), cfg_optim.GRAD_NORM_CLIP)
                    scaler.step(self.optimizer)
                    scaler.update()
                    accumulated_iter += 1
                else:
                    if loss == 0.:
                        pass
                        # print('loss is 0.') # No objs for all samples
                    elif torch.isfinite(loss):
                        loss.backward()
                        if self.grad_norm_clip is not None and self.grad_norm_clip > 0:
                            clip_grad_norm_(self.network.parameters(), self.grad_norm_clip)
                    else:
                        print('* Exception error (pipeline): nan or inf loss happend')
                        print('* Meta: ', dict_datum['meta'])

                    self.optimizer.step()
                    if not (self.scheduler is None):
                        self.scheduler.step()
                
                self.optimizer.zero_grad()
                mem_postfix = self._get_training_memory_postfix()
                if mem_postfix:
                    train_bar.set_postfix(mem_postfix)

                dict_logging = dict_net.get('logging', {}) if isinstance(dict_net, dict) else {}
                # iteration 단위 총 loss도 로깅
                dict_logging['loss'] = log_avg_loss
                idx_log_iter += 1
                wandb_metrics = {}

                if self.is_logging:
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (self.scheduler is None):
                        if self.optim_fastai:
                            lr = float(self.optimizer.lr)
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                        else:
                            lr = self.scheduler.get_last_lr()
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)

                if self.use_wandb:
                    wandb_metrics = {f'train/{k}': self._to_wandb_scalar(v) for k, v in dict_logging.items()}
                    if not (self.scheduler is None):
                        if self.optim_fastai:
                            lr_to_log = float(self.optimizer.lr)
                        else:
                            lr = self.scheduler.get_last_lr()
                            lr_to_log = float(lr[0]) if isinstance(lr, (list, tuple)) else float(lr)
                        wandb_metrics['train/learning_rate'] = lr_to_log
                    if not wandb_metrics:
                        wandb_metrics = {
                            'train/loss': float(log_avg_loss),
                        }
                    wandb_metrics.setdefault('iteration', idx_log_iter)
                    wandb_metrics.setdefault('train/iter', idx_log_iter)
                    wandb_metrics.setdefault('epoch', epoch)
                    self._wandb_log(wandb_metrics, step=idx_log_iter + self.wandb_iter_offset)
                # free memory (Killed error, checked with htop)
                if 'pointer' in dict_datum.keys():
                    for dict_item in dict_datum['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None
                for temp_key in dict_datum.keys():
                    dict_datum[temp_key] = None

            avg_loss_epoch = float(np.mean(avg_loss))

            path_model_last = None
            path_util_last = None
            path_model_best = None
            path_util_best = None
            dict_util = None

            if self.is_save_model:
                path_model_last = os.path.join(self.path_log, 'models', 'model_last.pt')
                path_util_last = os.path.join(self.path_log, 'utils', 'util_last.pt')
                path_model_best = os.path.join(self.path_log, 'models', 'model_best.pt')
                path_util_best = os.path.join(self.path_log, 'utils', 'util_best.pt')

                model_state = self.network.state_dict()
                dict_util = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'idx_log_iter': idx_log_iter,
                    'best_metric': self.best_metric,
                    'best_metric_name': self.best_metric_name,
                }
                if self.optim_fastai:
                    dict_util.update({'it': accumulated_iter})
                else:
                    if not (self.scheduler is None):
                        dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})

                # Save last
                _torch_save_atomic(model_state, path_model_last)
                _torch_save_atomic(dict_util, path_util_last)

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', avg_loss_epoch, epoch)
            if self.use_wandb:
                self._wandb_log({'train/avg_loss_epoch': avg_loss_epoch, 'epoch': epoch}, step=idx_log_iter + self.wandb_iter_offset)

            val_bev_metric = None
            val_bev_metric_name = None
            if self.is_validate:
                # K-Radar mode validation (KITTI anchor-based)
                if self.is_consider_subset:
                    if ((epoch + 1) % self.val_per_epoch_subset) == 0:
                        subset_metric = self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                        if (subset_metric is not None) and len(self.list_val_conf_thr) > 0:
                            val_bev_metric = subset_metric
                            val_bev_metric_name = f'val_subset/bev_ap_mean_conf_{self.list_val_conf_thr[0]}'
                if ((epoch + 1) % self.val_per_epoch_full) == 0:
                    full_metric = self.validate_kitti(epoch, list_conf_thr=self.list_val_conf_thr)
                    if (full_metric is not None) and len(self.list_val_conf_thr) > 0:
                        val_bev_metric = full_metric
                        val_bev_metric_name = f'val_tot/bev_ap_mean_conf_{self.list_val_conf_thr[0]}'

            if self.is_save_model and (val_bev_metric is not None):
                if val_bev_metric > self.best_metric:
                    self.best_metric = val_bev_metric
                    self.best_metric_name = val_bev_metric_name
                    dict_util_best = {
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'idx_log_iter': idx_log_iter,
                        'best_metric': self.best_metric,
                        'best_metric_name': self.best_metric_name,
                    }
                    if self.optim_fastai:
                        dict_util_best.update({'it': accumulated_iter})
                    else:
                        if not (self.scheduler is None):
                            dict_util_best.update({'scheduler_state_dict': self.scheduler.state_dict()})

                    _torch_save_atomic(dict_util_best['model_state_dict'], path_model_best)
                    _torch_save_atomic(dict_util_best, path_util_best)
                    print(f'* Update best checkpoint: {self.best_metric_name} = {self.best_metric:.4f} at epoch {epoch}')

                if dict_util is not None and path_util_last is not None:
                    dict_util['best_metric'] = self.best_metric
                    dict_util['best_metric_name'] = self.best_metric_name
                    _torch_save_atomic(dict_util, path_util_last)

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        msg = self.network.load_state_dict(pt_dict_model, strict=is_strict)
        missing = list(getattr(msg, 'missing_keys', []))
        unexpected = list(getattr(msg, 'unexpected_keys', []))
        print(
            f"* Loaded detection model from {path_dict_model} "
            f"(strict={is_strict}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if (not is_strict) and (len(missing) > 0 or len(unexpected) > 0):
            print(f"  missing keys (first 20): {missing[:20]}")
            print(f"  unexpected keys (first 20): {unexpected[:20]}")

    def load_best_checkpoint_for_eval(self, is_strict=False):
        if not hasattr(self, 'path_log') or self.path_log is None:
            print('* Warning: logging path is not initialized. Use current in-memory model for evaluation.')
            return None

        path_model_best = os.path.join(self.path_log, 'models', 'model_best.pt')
        if os.path.exists(path_model_best):
            self.load_dict_model(path_model_best, is_strict=is_strict)
            return path_model_best

        path_model_last = os.path.join(self.path_log, 'models', 'model_last.pt')
        if os.path.exists(path_model_last):
            print(f'* Warning: best checkpoint not found at {path_model_best}. Falling back to last checkpoint.')
            self.load_dict_model(path_model_last, is_strict=is_strict)
            return path_model_last

        print('* Warning: no saved checkpoint found. Use current in-memory model for evaluation.')
        return None

    # V2
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        self.network.eval()
        
        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)

            ### Vis data ###
            pc_lidar = dict_datum['ldr64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append('Sedan')
                    # list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)
            
            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                # colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred

    # V2
    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.training=False
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        

        ### Check is_validate with small dataset ###
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='* Test (Subset): ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_val), desc='* Test (Total): ')
            log_header = 'val_tot'

        # NOTE: On this server, spawning many DataLoader workers can cause slowdowns and
        # occasional segfaults at shutdown. Default to single-process loading for validation.
        try:
            val_num_workers = int(os.environ.get('KRADAR_VAL_NUM_WORKERS', '0'))
        except Exception:
            val_num_workers = 0
        if val_num_workers < 0:
            val_num_workers = int(self.cfg.OPTIMIZER.NUM_WORKERS)

        data_loader = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=1,
                shuffle=is_shuffle,
                collate_fn=self.dataset_val.collate_fn,
                num_workers=val_num_workers,
        )
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        # print(path_dir)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')
            f.close()

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break
            
            dict_out = None
            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except Exception as e:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                try:
                    print('* Exception: ', repr(e))
                except Exception:
                    pass
                try:
                    meta = dict_datum.get('meta', None) if isinstance(dict_datum, dict) else None
                    print('* Meta info: ', meta)
                except Exception:
                    pass
                is_feature_inferenced = False

            idx_name = str(idx_datum).zfill(6)

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)                
                if dict_out is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out = dict_datum_to_kitti(self, dict_out)

                if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out['kitti_gt']):
                        open_mode = 'w' if idx_label == 0 else 'a'
                        with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(label+'\n')
                    ### Gt ###

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out['kitti_desc'])
                    ### Process description ###

                    ### Pred: do not care len 0 with if else: already care as dummy ###
                    for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                        open_mode = 'w' if idx_pred == 0 else 'a'
                        with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(pred+'\n')
                    ### Pred: do not care len 0 with if else: already care as dummy ###

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
            
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        metrics_by_conf = {}

        ### Validate per conf ###
        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)

            list_metrics = []
            for idx_cls_val in self.list_val_care_idx:
                if self.is_validation_updated:
                    # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                    # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                    dict_metrics, result_log = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                else:
                    dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                print(f'-----conf{conf_thr}-----')
                print(result_log)
                list_metrics.append(dict_metrics)

            metrics_by_conf[conf_thr] = list_metrics

            for dict_metrics in list_metrics:
                cls_name = dict_metrics['cls']
                ious = dict_metrics['iou']
                bevs = dict_metrics['bev']
                ap3ds = dict_metrics['3d']
                self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                }, epoch)
                self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                }, epoch)
                if self.use_wandb:
                    self.wandb.log({
                        f'val/BEV_{conf_thr}_{cls_name}_{ious[0]}': bevs[0],
                        f'val/BEV_{conf_thr}_{cls_name}_{ious[1]}': bevs[1],
                        f'val/BEV_{conf_thr}_{cls_name}_{ious[2]}': bevs[2],
                        f'val/3D_{conf_thr}_{cls_name}_{ious[0]}': ap3ds[0],
                        f'val/3D_{conf_thr}_{cls_name}_{ious[1]}': ap3ds[1],
                        f'val/3D_{conf_thr}_{cls_name}_{ious[2]}': ap3ds[2],
                        'epoch': epoch
                    })
        ### Validate per conf ###

        ### Save validation results to text file ###
        self._save_validation_results_to_file(path_dir, epoch, is_subset, metrics_by_conf)
        ### Save validation results to text file ###

        if not list_conf_thr:
            return None

        plot_conf_thr = self._select_bev_ap_plot_conf(list_conf_thr, metrics_by_conf)
        plot_metric = self._mean_bev_ap_at_iou_from_metrics(
            metrics_by_conf.get(plot_conf_thr, []),
            target_iou=self.bev_ap_plot_iou,
        )
        self._record_bev_ap_plot_point(
            epoch=epoch,
            split_name='subset' if is_subset else 'total',
            conf_thr=plot_conf_thr,
            bev_ap=plot_metric,
        )

        primary_conf_thr = list_conf_thr[0]
        selection_metric = self._mean_bev_ap_from_metrics(metrics_by_conf.get(primary_conf_thr, []))
        if selection_metric is None:
            return None

        print(f'* Validation selection metric (BEV AP mean @ conf {primary_conf_thr}) = {selection_metric:.4f}')
        if (epoch is not None) and self.is_logging:
            self.log_test.add_scalar(
                f'{log_header}/best_selection_bev_ap_conf_{primary_conf_thr}',
                selection_metric,
                epoch,
            )
        if (epoch is not None) and self.use_wandb:
            self._wandb_log({
                f'{log_header}/best_selection_bev_ap_conf_{primary_conf_thr}': selection_metric,
                'epoch': epoch,
            })

        return selection_metric

    def _save_validation_results_to_file(self, path_dir, epoch, is_subset, metrics_by_conf):
        """
        Save validation results to a text file.

        Args:
            path_dir: Directory where validation results are stored
            epoch: Epoch number (None for final evaluation)
            is_subset: Whether this is subset validation
            metrics_by_conf: Dictionary of metrics organized by confidence threshold
        """
        try:
            # Determine output filename
            if is_subset:
                filename = f'validation_results_epoch_{epoch}_subset.txt' if epoch is not None else 'validation_results_subset.txt'
            else:
                filename = f'validation_results_epoch_{epoch}_total.txt' if epoch is not None else 'validation_results_total.txt'

            filepath = os.path.join(path_dir, filename)

            with open(filepath, 'w') as f:
                # Write header
                f.write(f"{'='*100}\n")
                f.write(f"Validation Results {'(Subset)' if is_subset else '(Full)'}\n")
                if epoch is not None:
                    f.write(f"Epoch: {epoch}\n")
                f.write(f"{'='*100}\n\n")

                # Write results for each confidence threshold
                for conf_thr in sorted(metrics_by_conf.keys()):
                    list_metrics = metrics_by_conf[conf_thr]
                    f.write(f"\n{'─'*100}\n")
                    f.write(f"Confidence Threshold: {conf_thr}\n")
                    f.write(f"{'─'*100}\n\n")

                    # Table header
                    f.write(f"{'Class':<15} {'IoU':<10} {'BEV AP':<15} {'3D AP':<15}\n")
                    f.write(f"{'-'*55}\n")

                    # Write metrics for each class
                    for dict_metrics in list_metrics:
                        cls_name = dict_metrics['cls']
                        ious = dict_metrics['iou']
                        bevs = dict_metrics['bev']
                        ap3ds = dict_metrics['3d']

                        for idx_iou in range(len(ious)):
                            iou_val = ious[idx_iou]
                            bev_val = bevs[idx_iou]
                            ap3d_val = ap3ds[idx_iou]

                            if idx_iou == 0:
                                f.write(f"{cls_name:<15} {iou_val:<10.2f} {bev_val:<15.2f} {ap3d_val:<15.2f}\n")
                            else:
                                f.write(f"{'':<15} {iou_val:<10.2f} {bev_val:<15.2f} {ap3d_val:<15.2f}\n")

                    f.write("\n")

                # Write summary statistics
                f.write(f"\n{'='*100}\n")
                f.write("Summary Statistics\n")
                f.write(f"{'='*100}\n\n")

                if metrics_by_conf:
                    primary_conf = sorted(metrics_by_conf.keys())[0]
                    list_metrics = metrics_by_conf[primary_conf]

                    for dict_metrics in list_metrics:
                        cls_name = dict_metrics['cls']
                        bevs = dict_metrics['bev']
                        ap3ds = dict_metrics['3d']

                        avg_bev = np.mean(bevs)
                        avg_ap3d = np.mean(ap3ds)

                        f.write(f"Class: {cls_name}\n")
                        f.write(f"  BEV AP (mean):     {avg_bev:.4f}%\n")
                        f.write(f"  3D AP (mean):      {avg_ap3d:.4f}%\n")
                        f.write(f"  BEV AP@0.7:        {bevs[0]:.4f}%\n")
                        f.write(f"  BEV AP@0.5:        {bevs[1]:.4f}%\n")
                        f.write(f"  BEV AP@0.3:        {bevs[2]:.4f}%\n")
                        f.write(f"  3D AP@0.7:         {ap3ds[0]:.4f}%\n")
                        f.write(f"  3D AP@0.5:         {ap3ds[1]:.4f}%\n")
                        f.write(f"  3D AP@0.3:         {ap3ds[2]:.4f}%\n\n")

                f.write(f"{'='*100}\n")

            print(f"✓ Validation results saved to: {filepath}")

        except Exception as e:
            print(f"⚠ Warning: Failed to save validation results to file. Error: {str(e)}")

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        
        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        # Hold-out test split (test.txt). validate_kitti_conditional is the post-training
        # final eval: iterate dataset_test, not dataset_val.
        eval_dataset = self.dataset_test

        # Check is_validate with small dataset
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(eval_dataset), desc='Test (Total): ')

        try:
            val_num_workers = int(os.environ.get('KRADAR_VAL_NUM_WORKERS', '0'))
        except Exception:
            val_num_workers = 0
        if val_num_workers < 0:
            val_num_workers = int(self.cfg.OPTIMIZER.NUM_WORKERS)

        data_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=is_shuffle,
                collate_fn=eval_dataset.collate_fn,
                num_workers=val_num_workers,
        )
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            ### For All Conditions ###
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)
                            
            ### For Specific Conditions ###
            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)
                
                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        # Creating gts and preds txt files for evaluation
        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            if is_print_memory:
                mem_postfix = self._get_training_memory_postfix()
                if mem_postfix:
                    tqdm_bar.set_postfix(mem_postfix)
                
            idx_name = str(idx_datum).zfill(6)
            
            road_cond_tag, time_cond_tag, weather_cond_tag = \
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
            # print(dict_out['desc'][0])

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)
                
                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    ### Process description ###
                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                    
                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)
                        
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
        for conf_thr in list_conf_thr:
            for condition in all_condition_list:
                try:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    val_ids = read_imageset_file(split_path)
                    if len(val_ids) == 0:
                        print(f'* Skip condition: no samples (conf_thr={conf_thr}, condition={condition})')
                        continue

                    dt_annos = kitti.get_label_annos(preds_dir)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        if self.is_validation_updated:
                            # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                            # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                            dict_metrics, result = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        else:
                            dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)
                            
                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
                except Exception as e:
                    print(f'* Exception error (Pipeline): condition eval failed '
                          f'(conf_thr={conf_thr}, condition={condition}): {repr(e)}')

        for conf_thr in list_conf_thr:
            path_check = os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt')
            print(f'* Check {path_check}')
        ### Validate per conf ###
