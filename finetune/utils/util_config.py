"""
# source: https://github.com/open-mmlab/OpenPCDet/blob/1f5b7872b03e9e3d42801872bc59681ef36357b5/pcdet/config.py
"""

import yaml
from easydict import EasyDict
from pathlib import Path
import os


def _expandvars_and_user(value):
    if not isinstance(value, str):
        return value
    # Expand ~ and $VARS. If nothing to expand, this is a no-op.
    return os.path.expandvars(os.path.expanduser(value))


def _maybe_resolve_relative_path(value, base_dir: Path):
    if not isinstance(value, str):
        return value

    expanded = _expandvars_and_user(value)
    # Avoid touching URLs.
    if "://" in expanded:
        return expanded

    p = Path(expanded)
    if p.is_absolute():
        return str(p)

    # Only resolve strings that look like paths (contain a separator or start with dot).
    looks_like_path = ("/" in expanded) or ("\\" in expanded) or expanded.startswith(".")
    if not looks_like_path:
        return expanded

    return str((base_dir / p).resolve())


def _walk_and_fix(obj, base_dir: Path):
    if isinstance(obj, dict):
        return {k: _walk_and_fix(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _walk_and_fix(v, base_dir) for v in obj ]
    if isinstance(obj, str):
        return _maybe_resolve_relative_path(obj, base_dir)
    return obj

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        # Resolve base config relative to the current config file (set by cfg_from_yaml_file).
        base_dir = getattr(config, "_CFG_DIR", None)
        base_cfg_path = new_config["_BASE_CONFIG_"]
        if base_dir is not None:
            expanded = _expandvars_and_user(str(base_cfg_path))
            p = Path(expanded)
            if not p.is_absolute():
                base_cfg_path = str((Path(base_dir) / p).resolve())
            else:
                base_cfg_path = str(p)
        with open(base_cfg_path, 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        project_root = Path(getattr(config, "_PROJECT_ROOT", Path(base_cfg_path).resolve().parent)).resolve()
        yaml_config = _walk_and_fix(yaml_config, project_root)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    cfg_path = Path(_expandvars_and_user(str(cfg_file))).resolve()
    # Keep the config directory so _BASE_CONFIG_ can be resolved relative to it.
    config._CFG_DIR = str(cfg_path.parent)
    try:
        project_root = Path(config.ROOT_DIR).resolve()
    except Exception:
        project_root = cfg_path.parent
    config._PROJECT_ROOT = str(project_root)

    with open(cfg_path, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        new_config = _walk_and_fix(new_config, project_root)
        merge_new_config(config=config, new_config=new_config)

    return config

# set empty as default and add key & values
cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
