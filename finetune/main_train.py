'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

'''
Single-stage fine-tuning for MAE-pretrained radar object detection.

Adapted from kaist-avelab/K-Radar (https://github.com/kaist-avelab/K-Radar).
Wraps the K-Radar PipelineDetection_v1_0 to drive an MAE-initialized backbone
through a single fine-tuning + final test phase. The underlying pipeline,
datasets, models, and utilities under finetune/ retain their original AVELab
attribution.
'''

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_CFG = Path(__file__).resolve().parent / 'configs' / 'cfg_RadarMAE.yml'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / 'outputs' / 'finetune'
LOG_DIR = RUN_DIR / 'logs'
EFFECTIVE_CFG_DIR = RUN_DIR / 'effective_configs'
K_RADAR_V1_LABEL_ROOT = PROJECT_ROOT / 'data' / 'K-Radar_RA' / 'RAE'


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self.streams[0].isatty()
        except Exception:
            return False


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]", "_", name)


def _derive_tag_from_cfg(cfg_path: str):
    """Derive a tag from the cfg's PRETRAIN_PATH. Returns (tag, ckpt_path_str_or_None)."""
    try:
        import yaml
        with open(cfg_path) as f:
            yml = yaml.safe_load(f)
        ckpt = (yml.get('MODEL', {}) or {}).get('BACKBONE', {}).get('MAE', {}).get('PRETRAIN_PATH', '')
        ckpt = ckpt.strip() if isinstance(ckpt, str) else ''
    except Exception:
        ckpt = ''
    if ckpt:
        p = (Path(__file__).resolve().parent / ckpt).resolve()  # cfg paths are relative to finetune/
        try:
            return _sanitize(f"{p.parent.parent.name}__{p.parent.name}__{p.stem}"), ckpt
        except Exception:
            return _sanitize(p.stem), ckpt
    return _sanitize(Path(cfg_path).stem), None


def setup_direct_run_logging(cfg_path: str):
    """When running directly, mirror stdout/stderr into _finetune_runs_neurips_v3/logs/.

    When invoked through a launcher (run_finetune_all_ckpts.py), the launcher already
    sets FINETUNE_BEV_AP_PLOT_PATH, so skip in that case.
    """
    if os.environ.get('FINETUNE_BEV_AP_PLOT_PATH', '').strip():
        return None  # launcher mode

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag, ckpt = _derive_tag_from_cfg(cfg_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f'direct_{timestamp}_{tag}.log'
    plot_file = LOG_DIR / f'direct_{timestamp}_{tag}.bev_ap_iou0p3.png'

    os.environ['FINETUNE_BEV_AP_PLOT_PATH'] = str(plot_file)
    os.environ.setdefault('FINETUNE_BEV_AP_PLOT_IOU', '0.3')
    os.environ.setdefault('FINETUNE_BEV_AP_PLOT_CONF', '0.3')

    log_fh = open(log_file, 'w', buffering=1)
    log_fh.write(f"[start] {datetime.now().isoformat()}\n")
    log_fh.write(f"[cfg]   {cfg_path}\n")
    log_fh.write(f"[ckpt]  {ckpt or '<none>'}\n")
    log_fh.write(f"[plot]  {plot_file}\n")
    log_fh.write(f"[cwd]   {os.getcwd()}\n")
    log_fh.write(f"[cmd]   {' '.join(sys.argv)}\n")
    log_fh.write(f"[env]   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}\n\n")
    log_fh.flush()

    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)
    print(f"[main_train] direct-run log: {log_file}")
    print(f"[main_train] direct-run plot: {plot_file}")
    return log_fh


def _restrict_classes(cfg, keep):
    """Restrict detection to `keep` (ordered list of K-Radar class names) so the
    head builds exactly len(keep) class(es). Rewrites DATASET.label /
    DATASET.CLASS_INFO, MODEL.HEAD.ANCHOR_GENERATOR_CONFIG (single-class NMS when
    one class remains) and VAL.LIST_CARE_VAL. Class order follows the existing
    ANCHOR_GENERATOR_CONFIG. No-op when `keep` is empty."""
    if not keep:
        return
    dataset = cfg['DATASET']
    head = cfg['MODEL']['HEAD']
    anchors = head['ANCHOR_GENERATOR_CONFIG']
    canonical = [a['class_name'] for a in anchors]
    unknown = [c for c in keep if c not in canonical]
    if unknown:
        raise ValueError(f"--classes: unknown class(es) {unknown}; valid options: {canonical}")
    keep_set = set(keep)
    kept = [c for c in canonical if c in keep_set]        # preserve canonical order

    # anchors -> kept classes only (same order)
    head['ANCHOR_GENERATOR_CONFIG'] = [a for a in anchors if a['class_name'] in keep_set]
    # a single remaining class REQUIRES class-agnostic NMS: multi-class NMS builds
    # an empty label mapping (arange(1, 1)) and crashes when num_class == 1.
    if len(kept) == 1:
        nms = head.setdefault('POST_PROCESSING', {}).setdefault('NMS_CONFIG', {})
        nms['MULTI_CLASSES_NMS'] = False

    new_id = {name: i + 1 for i, name in enumerate(kept)}  # contiguous ids 1..K

    # DATASET.label entries are [enabled, logit_idx, rgb, bgr]; the head + evaluator
    # build class_names from entries with logit_idx > 0, and the dataset keeps GT
    # whose class is enabled -> disable dropped classes on both counts.
    lab = dataset['label']
    for name in canonical:
        if name in lab:
            lab[name][0] = name in keep_set
            lab[name][1] = new_id.get(name, -1)

    # CLASS_INFO (used by some heads / vis; kept consistent)
    ci = dataset.setdefault('CLASS_INFO', {})
    ci['CLASS_ID'] = {**{n: new_id[n] for n in kept}, 'Background': 0}
    ci['NUM_CLS'] = len(kept)

    # evaluate only the kept classes
    cfg['VAL']['LIST_CARE_VAL'] = list(kept)
    print(f"* Class restriction: keeping {kept} (num_class={len(kept)})")


def prepare_v1_0_label_cfg(cfg_path: str, scratch: bool = False,
                           pretrain_path: str = None,
                           seed: int = None,
                           degrade_axis: str = None, degrade_factor: int = 2,
                           keep_classes=None,
                           max_epoch: int = None,
                           exp_name: str = None,
                           tokenizer: str = None,
                           mae_patch_size: str = None,
                           pos_embed_mode: str = None) -> str:
    """Create an effective cfg that uses K-Radar v1.0 info_label files."""
    import yaml

    cfg_path = Path(cfg_path).expanduser().resolve()
    if not K_RADAR_V1_LABEL_ROOT.is_dir():
        raise FileNotFoundError(f'K-Radar v1.0 label root not found: {K_RADAR_V1_LABEL_ROOT}')

    with open(cfg_path) as f:
        yml = yaml.safe_load(f)

    if not isinstance(yml, dict):
        raise ValueError(f'Invalid yaml config: {cfg_path}')

    if seed is not None:
        general = yml.setdefault('GENERAL', {})
        if not isinstance(general, dict):
            raise ValueError(f'Invalid GENERAL section in config: {cfg_path}')
        general['SEED'] = int(seed)

    if exp_name:
        general = yml.setdefault('GENERAL', {})
        if not isinstance(general, dict):
            raise ValueError(f'Invalid GENERAL section in config: {cfg_path}')
        general['NAME'] = str(exp_name)

    if max_epoch is not None:
        optimizer = yml.setdefault('OPTIMIZER', {})
        if not isinstance(optimizer, dict):
            raise ValueError(f'Invalid OPTIMIZER section in config: {cfg_path}')
        optimizer['MAX_EPOCH'] = int(max_epoch)
        # VAL_PER_EPOCH_FULL is a modulo period: leaving it at the original
        # (longer) MAX_EPOCH means a short run never triggers a periodic full
        # validation at all. Keep it tied to the shortened horizon.
        val = yml.setdefault('VAL', {})
        if isinstance(val, dict) and 'VAL_PER_EPOCH_FULL' in val:
            val['VAL_PER_EPOCH_FULL'] = min(int(val['VAL_PER_EPOCH_FULL']), int(max_epoch))

    dataset = yml.setdefault('DATASET', {})
    if not isinstance(dataset, dict):
        raise ValueError(f'Invalid DATASET section in config: {cfg_path}')

    if scratch:
        try:
            mae_cfg = yml['MODEL']['BACKBONE']['MAE']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Missing MODEL.BACKBONE.MAE section in config: {cfg_path}') from exc
        mae_cfg['PRETRAIN_PATH'] = ''
        mae_cfg['AUTO_OVERRIDE_FROM_PRETRAIN'] = False
    elif pretrain_path:
        try:
            mae_cfg = yml['MODEL']['BACKBONE']['MAE']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Missing MODEL.BACKBONE.MAE section in config: {cfg_path}') from exc
        resolved_pretrain_path = Path(pretrain_path).expanduser().resolve()
        if not resolved_pretrain_path.is_file():
            raise FileNotFoundError(f'Pretraining checkpoint not found: {resolved_pretrain_path}')
        mae_cfg['PRETRAIN_PATH'] = str(resolved_pretrain_path)

    # Encoder-geometry overrides.  The tokenization ablation swaps the encoder
    # between axial stripes and square patches, so the finetune backbone has to
    # be built to match the checkpoint.  Only these three keys ever differ --
    # IMG_HEIGHT/WIDTH and every other MAE key stay at the config's values.
    if tokenizer or mae_patch_size or pos_embed_mode:
        try:
            mae_cfg = yml['MODEL']['BACKBONE']['MAE']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Missing MODEL.BACKBONE.MAE section in config: {cfg_path}') from exc
        if tokenizer:
            mae_cfg['TOKENIZER'] = str(tokenizer)
        if mae_patch_size:
            parts = [int(v) for v in str(mae_patch_size).replace('x', ',').split(',') if str(v).strip()]
            if len(parts) not in (1, 2):
                raise ValueError(f'--mae-patch-size must be N or H,W; got {mae_patch_size!r}')
            mae_cfg['PATCH_SIZE'] = parts * 2 if len(parts) == 1 else parts
        if pos_embed_mode:
            mae_cfg['POS_EMBED_MODE'] = str(pos_embed_mode).upper()
        print(f"* MAE encoder override: TOKENIZER={mae_cfg.get('TOKENIZER')}, "
              f"PATCH_SIZE={mae_cfg.get('PATCH_SIZE')}, POS_EMBED_MODE={mae_cfg.get('POS_EMBED_MODE')}")

    # Optional per-axis RA-map resolution degradation (applies to train + eval).
    if degrade_axis:
        ra_map_cfg = dataset.setdefault('ra_map', {})
        if isinstance(ra_map_cfg, dict):
            ra_map_cfg['degrade'] = {'axis': degrade_axis, 'factor': int(degrade_factor)}

    # Dataset v2.x selects labels by DATASET.label_version.
    # v1_0 maps to <list_dir_kradar>/<seq>/info_label/*.txt.
    path_data = dataset.setdefault('path_data', {})
    if isinstance(path_data, dict):
        path_data['list_dir_kradar'] = [str(K_RADAR_V1_LABEL_ROOT)]
        dataset['label_version'] = 'v1_0'

    # Keep old v1.x dataset configs aligned if a caller passes that schema.
    dir_cfg = dataset.setdefault('DIR', {})
    if isinstance(dir_cfg, dict):
        dir_cfg['LIST_DIR'] = [str(K_RADAR_V1_LABEL_ROOT)]

    label_cfg = dataset.get('LABEL')
    if isinstance(label_cfg, dict) and 'IS_USE_REVISED_LABEL' in label_cfg:
        label_cfg['IS_USE_REVISED_LABEL'] = False

    # Optional single-/subset-class restriction (e.g. Sedan-only finetuning).
    _restrict_classes(yml, keep_classes)

    EFFECTIVE_CFG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    effective_cfg = EFFECTIVE_CFG_DIR / f'{cfg_path.stem}_v1_0_info_label_{timestamp}_{os.getpid()}.yml'
    with open(effective_cfg, 'w') as f:
        yaml.safe_dump(yml, f, sort_keys=False, default_flow_style=False)

    return str(effective_cfg)


# Parameter groups that MAEPolarBEVBackbone._load_pretrained_encoder() fills from
# PRETRAIN_PATH. Freezing exactly this set = "linear probe on the pretrained encoder":
# everything the pretraining produced is held fixed, everything else is trained.
ENCODER_PREFIXES_STRIPE = ('input_norm', 'conv_stem', 'stripe_embed', 'axis_embed',
                           'geome_pe', 'blocks', 'norm')
ENCODER_PREFIXES_PATCH = ('input_norm', 'conv_stem', 'patch_proj', 'patch_embed',
                          'encoder_pos_embed', 'blocks', 'norm')


def _pin_module_to_eval(module):
    """Make module.train(True) a no-op that keeps the module in eval mode.

    requires_grad=False alone does not freeze a module: BatchNorm running stats keep
    adapting in train mode, and train_network() calls network.train() at every epoch,
    which would undo a one-shot .eval().
    """
    import torch.nn as nn

    def _train(mode=True, _m=module):
        nn.Module.train(_m, False)   # unbound call -> no recursion into this override
        return _m

    nn.Module.train(module, False)
    module.train = _train


class _ParamsProxy:
    """Minimal stand-in for a model so build_optimizer() sees only trainable params."""

    def __init__(self, params):
        self._params = list(params)

    def parameters(self):
        return self._params


def freeze_pretrained_encoder(pline):
    """Freeze the pretrained MAE encoder; train only BEV projection + detection head.

    Also rebuilds the optimizer/scheduler: the pipeline constructor already built the
    optimizer over *all* parameters, so frozen tensors would otherwise still sit in a
    param group (and keep collecting weight decay / Adam state).
    """
    import torch.nn as nn
    from utils.util_pipeline import build_optimizer, build_scheduler

    network = pline.network
    backbone = getattr(network, 'backbone', None)
    if backbone is None:
        raise RuntimeError('--freeze-encoder: network has no backbone module')

    tokenizer = str(getattr(backbone, 'tokenizer', 'stripe'))
    prefixes = ENCODER_PREFIXES_STRIPE if tokenizer == 'stripe' else ENCODER_PREFIXES_PATCH

    frozen_names, frozen_bn = [], []
    n_frozen = 0
    for name, module in backbone.named_children():
        if not name.startswith(prefixes):
            continue
        for param in module.parameters():
            param.requires_grad_(False)
            n_frozen += param.numel()
        _pin_module_to_eval(module)
        frozen_names.append(name)
        for sub_name, sub in module.named_modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                frozen_bn.append(f'{name}.{sub_name}'.rstrip('.'))

    # Parameters held directly on the backbone (axis_embed, encoder_pos_embed) are not
    # covered by named_children().
    for name, param in backbone.named_parameters(recurse=False):
        if name.startswith(prefixes):
            param.requires_grad_(False)
            n_frozen += param.numel()
            frozen_names.append(f'{name} (parameter)')

    trainable = [p for p in network.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = n_frozen + n_trainable

    pline.optimizer = build_optimizer(pline, _ParamsProxy(trainable))
    pline.scheduler = build_scheduler(pline, pline.optimizer)

    in_optim = {id(p) for group in pline.optimizer.param_groups for p in group['params']}
    leaked = [p for p in network.parameters() if not p.requires_grad and id(p) in in_optim]
    if leaked:
        raise RuntimeError(f'--freeze-encoder: {len(leaked)} frozen tensors are still in the optimizer')
    if len(in_optim) != len(trainable):
        raise RuntimeError(
            f'--freeze-encoder: optimizer holds {len(in_optim)} tensors, expected {len(trainable)}')

    # network.train() must not resurrect the frozen modules.
    network.train()
    still_training = [name for name, module in backbone.named_children()
                      if name.startswith(prefixes) and module.training]
    if still_training:
        raise RuntimeError(f'--freeze-encoder: modules back in train mode after network.train(): {still_training}')

    print('=' * 60)
    print('* --freeze-encoder: pretrained encoder is frozen (linear-probe mode)')
    print(f'*   frozen modules   : {frozen_names}')
    print(f'*   frozen BatchNorm : {frozen_bn} (pinned to eval, running stats fixed)')
    print(f'*   frozen params    : {n_frozen/1e6:.2f}M')
    print(f'*   trainable params : {n_trainable/1e6:.2f}M ({100.0*n_trainable/max(n_total,1):.1f}% of {n_total/1e6:.2f}M)')
    print(f'*   optimizer tensors: {len(in_optim)} (rebuilt after freezing)')
    print('=' * 60)


def load_last_checkpoint_for_eval(pline, is_strict=False):
    path_model_last = Path(pline.path_log) / 'models' / 'model_last.pt'
    if not path_model_last.exists():
        try:
            resume_cfg = pline.cfg.GENERAL.RESUME
            resume_path = Path(resume_cfg.PATH_EXP) / 'models' / 'model_last.pt'
            if resume_cfg.IS_RESUME and resume_path.exists():
                path_model_last = resume_path
            else:
                raise FileNotFoundError
        except Exception as exc:
            raise FileNotFoundError(f'Last checkpoint not found for evaluation: {path_model_last}') from exc

    pline.load_dict_model(str(path_model_last), is_strict=is_strict)
    print(f"[main_train] evaluation checkpoint: {path_model_last}")
    return str(path_model_last)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=str(DEFAULT_CFG), help='Path to config yaml')
    parser.add_argument(
        '--scratch',
        action='store_true',
        help='Train with a randomly initialized backbone instead of PRETRAIN_PATH.',
    )
    parser.add_argument(
        '--pretrain-path',
        type=str,
        default='',
        help='Override MODEL.BACKBONE.MAE.PRETRAIN_PATH from the source config.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override GENERAL.SEED from the source config.',
    )
    parser.add_argument(
        '--degrade-axis',
        type=str,
        default='none',
        choices=['none', 'range', 'azimuth'],
        help="Halve the RA-map resolution along this axis for train + eval. Default: none.",
    )
    parser.add_argument(
        '--degrade-factor',
        type=int,
        default=2,
        help='Resolution degradation factor along --degrade-axis (default 2 = 2x).',
    )
    parser.add_argument(
        '--classes',
        type=str,
        default='',
        help="Comma-separated K-Radar class name(s) to finetune on (e.g. 'Sedan'). "
             "Empty = all 7 classes.",
    )
    parser.add_argument(
        '--max-epoch',
        type=int,
        default=None,
        help='Override OPTIMIZER.MAX_EPOCH (also caps VAL.VAL_PER_EPOCH_FULL).',
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true',
        help='Freeze the pretrained MAE encoder (weights + BatchNorm stats) and train '
             'only the BEV projection and detection head.',
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        default='',
        help='Override GENERAL.NAME, i.e. the exp_<time>_<NAME> log directory suffix.',
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='',
        choices=['', 'stripe', 'patch'],
        help='Override MODEL.BACKBONE.MAE.TOKENIZER to match the pretrained checkpoint.',
    )
    parser.add_argument(
        '--mae-patch-size',
        type=str,
        default='',
        help="Override MODEL.BACKBONE.MAE.PATCH_SIZE (e.g. '8,8' or '4,16').",
    )
    parser.add_argument(
        '--pos-embed-mode',
        type=str,
        default='',
        choices=['', 'PPE', 'PE'],
        help='Override MODEL.BACKBONE.MAE.POS_EMBED_MODE to match the pretrained checkpoint.',
    )
    args = parser.parse_args()

    degrade_axis = None if args.degrade_axis == 'none' else args.degrade_axis
    keep_classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    effective_cfg = prepare_v1_0_label_cfg(
        args.cfg, scratch=args.scratch, pretrain_path=args.pretrain_path,
        seed=args.seed,
        degrade_axis=degrade_axis, degrade_factor=args.degrade_factor,
        keep_classes=keep_classes,
        max_epoch=args.max_epoch, exp_name=args.exp_name,
        tokenizer=args.tokenizer or None,
        mae_patch_size=args.mae_patch_size or None,
        pos_embed_mode=args.pos_embed_mode or None,
    )
    log_fh = setup_direct_run_logging(effective_cfg)
    print(f"[main_train] source cfg: {Path(args.cfg).expanduser().resolve()}")
    print(f"[main_train] effective cfg: {effective_cfg}")
    print(f"[main_train] initialization: {'scratch' if args.scratch else 'pretrained checkpoint'}")
    print(f"[main_train] encoder: {'frozen (probe)' if args.freeze_encoder else 'finetuned'}"
          f"{f', max_epoch={args.max_epoch}' if args.max_epoch is not None else ''}")
    print(f"[main_train] K-Radar labels: {K_RADAR_V1_LABEL_ROOT}/<seq>/info_label/*.txt")

    # Import after tee setup so import-time output is also captured in the log.
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    rc = 0
    try:
        pline = PipelineDetection_v1_0(path_cfg=effective_cfg, mode='train')

        ### Save this file for checking ###
        import shutil
        shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
        ### Save this file for checking ###

        if args.freeze_encoder:
            freeze_pretrained_encoder(pline)

        pline.train_network()
        load_last_checkpoint_for_eval(pline)
        final_val_epoch = int(pline.cfg.OPTIMIZER.MAX_EPOCH)
        final_test_epoch = final_val_epoch + 1

        # Final validation on the validation split using model_last.pt.
        if getattr(pline, 'is_validate', False):
            pline.validate_kitti(
                epoch=final_val_epoch,
                list_conf_thr=[0.3],
                is_subset=False,
            )

        # Final test evaluation on the test split using model_last.pt.
        pline.validate_kitti_conditional(
            epoch=final_test_epoch,
            list_conf_thr=[0.3],
            is_subset=False,
            is_print_memory=False,
        )
    except Exception:
        rc = 1
        raise
    finally:
        if log_fh is not None:
            try:
                log_fh.write(f"\n[end]   {datetime.now().isoformat()}  rc={rc}\n")
                log_fh.flush()
                log_fh.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
