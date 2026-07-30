"""Unified technique label configuration loader.

Single source of truth for tonal_technique / articulation / legato class
definitions, display names, ID remapping, and RWC technique mapping.

Configuration lives in:  <project_root>/config/technique_label_config.json

Remap example — to merge staccato into spiccato, edit the JSON:
    "articulation": {
        "classes": {"0": "none", "1": "release", "2": "staccato", "3": "spiccato"},
        "remap": {"staccato": "spiccato"}
    }
This causes every articulation ID 2 (staccato) to be replaced by ID 3
(spiccato) at data-loading time.  Training and evaluation both honour
the same config.
"""

import json
import os
import logging
import numpy as np

_logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'config', 'technique_label_config.json',
)

_TASK_KEYS = ('tonal_technique', 'articulation', 'legato')
_TASK_SHORT = {'tonal_technique': 'tonal', 'articulation': 'artic', 'legato': 'legato'}


class TechniqueLabels:
    """Parsed technique label configuration with remap support."""

    def __init__(self, config_path=None):
        path = config_path or _CONFIG_PATH
        path = os.path.abspath(path)
        with open(path) as f:
            self._cfg = json.load(f)
        self._path = path
        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tonal_names(self):
        """dict  {int_id: str_name} for tonal_technique."""
        return dict(self._names['tonal_technique'])

    @property
    def artic_names(self):
        """dict  {int_id: str_name} for articulation."""
        return dict(self._names['articulation'])

    @property
    def legato_names(self):
        """dict  {int_id: str_name} for legato."""
        return dict(self._names['legato'])

    @property
    def num_tonal_classes(self):
        return len(self._names['tonal_technique'])

    @property
    def num_artic_classes(self):
        return len(self._names['articulation'])

    @property
    def num_legato_classes(self):
        return len(self._names['legato'])

    @property
    def has_remap(self):
        return any(len(t) > 0 for t in self._id_remap.values())

    def remap_tonal(self, original_id):
        """Remap a single tonal_technique ID (int → int)."""
        return self._id_remap['tonal_technique'].get(int(original_id),
                                                     int(original_id))

    def remap_artic(self, original_id):
        """Remap a single articulation ID (int → int)."""
        return self._id_remap['articulation'].get(int(original_id),
                                                  int(original_id))

    def remap_legato(self, original_id):
        """Remap a single legato ID (int → int)."""
        return self._id_remap['legato'].get(int(original_id),
                                            int(original_id))

    def remap_array(self, task, arr):
        """Vectorised remap for a numpy int array.  Returns a new array."""
        table = self._id_remap.get(task, {})
        if not table:
            return arr
        out = arr.copy()
        for src, dst in table.items():
            out[arr == src] = dst
        return out

    @property
    def rwc_technique_map(self):
        """dict  {technique_name: {'tonal': int, 'artic': int, 'legato': int}}
        with remap already applied."""
        return dict(self._rwc_map)

    def describe(self):
        """Print a human-readable summary of the current config."""
        lines = [f'TechniqueLabels  (from {self._path})', '']
        for task in _TASK_KEYS:
            names = self._names[task]
            remap = self._id_remap[task]
            lines.append(f'  {task}  ({len(names)} classes)')
            for k in sorted(names):
                tag = ''
                for src, dst in remap.items():
                    if dst == k:
                        src_name = self._names[task].get(src, f'?{src}')
                        tag = f'  ← (includes remapped {src_name})'
                lines.append(f'    {k}: {names[k]}{tag}')
            if remap:
                for src, dst in remap.items():
                    lines.append(f'    [remap] {self._raw_names[task][src]} '
                                 f'(id={src}) → {self._raw_names[task][dst]} '
                                 f'(id={dst})')
            lines.append('')
        lines.append('  rwc_technique_map:')
        for tech, m in sorted(self._rwc_map.items()):
            lines.append(f'    {tech:12s} → tonal={m["tonal"]}  '
                         f'artic={m["artic"]}  legato={m["legato"]}')
        print('\n'.join(lines))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self):
        self._names = {}       # task → {int_id: str_name}
        self._raw_names = {}   # same but before remap filtering
        self._id_remap = {}    # task → {src_id(int): dst_id(int)}
        self._name_to_id = {}  # task → {name: int_id}

        for task in _TASK_KEYS:
            task_cfg = self._cfg[task]
            classes = task_cfg['classes']          # {"0": "none", ...}
            remap_rules = task_cfg.get('remap', {})  # {"staccato": "spiccato"}

            name_to_id = {}
            id_to_name = {}
            for k, name in classes.items():
                int_k = int(k)
                name_to_id[name] = int_k
                id_to_name[int_k] = name

            self._raw_names[task] = dict(id_to_name)
            self._name_to_id[task] = name_to_id

            id_remap = {}
            for src_name, dst_name in remap_rules.items():
                if src_name not in name_to_id:
                    _logger.warning(
                        f'[TechniqueLabels] remap source "{src_name}" '
                        f'not found in {task} classes — skipping')
                    continue
                if dst_name not in name_to_id:
                    _logger.warning(
                        f'[TechniqueLabels] remap target "{dst_name}" '
                        f'not found in {task} classes — skipping')
                    continue
                src_id = name_to_id[src_name]
                dst_id = name_to_id[dst_name]
                if src_id == dst_id:
                    continue
                id_remap[src_id] = dst_id
                _logger.info(f'[TechniqueLabels] {task}: {src_name}(id={src_id})'
                             f' → {dst_name}(id={dst_id})')

            self._id_remap[task] = id_remap
            self._names[task] = id_to_name

        # RWC technique map (resolve names → IDs + apply remap)
        rwc_cfg = self._cfg.get('rwc_technique_map', {})
        self._rwc_map = {}
        for tech_name, mapping in rwc_cfg.items():
            tonal_id = self._resolve_and_remap(
                'tonal_technique', mapping.get('tonal', 'none'))
            artic_id = self._resolve_and_remap(
                'articulation', mapping.get('artic', 'none'))
            legato_id = self._resolve_and_remap(
                'legato', mapping.get('legato', 'bow_change'))
            self._rwc_map[tech_name] = {
                'tonal': tonal_id, 'artic': artic_id, 'legato': legato_id}

    def _resolve_and_remap(self, task, name):
        raw_id = self._name_to_id[task].get(name, 0)
        return self._id_remap[task].get(raw_id, raw_id)


# ======================================================================
# Module-level singleton
# ======================================================================

_instance = None


def get_technique_labels(config_path=None):
    """Return a (cached) TechniqueLabels instance.

    Call with no arguments in normal usage; all scripts share the same
    default config path.  Pass an explicit path only for testing.
    """
    global _instance
    if _instance is None or config_path is not None:
        _instance = TechniqueLabels(config_path)
    return _instance
