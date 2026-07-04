#!/usr/bin/env python3
"""Fix bugs in D1, D3, D2 patches."""
import py_compile

# Fix D1+D3: pop custom config from kwargs before super().__init__
HEAD = '/home/yelingfei/projects/GCD/mmdet/models/dense_heads/gdino_head_inc_gcd.py'
src = open(HEAD, encoding='utf-8').read()

old = '        super().__init__(**kwargs)'
new = ('        _dt_cfg_raw = kwargs.pop(\'distill_trunc_cfg\', None)\n'
       '        _gd_cfg_raw = kwargs.pop(\'gt_dup_cfg\', None)\n'
       '        super().__init__(**kwargs)')

if old in src and '_dt_cfg_raw' not in src:
    src = src.replace(old, new, 1)
    src = src.replace(
        "_dt_cfg = getattr(self, 'distill_trunc_cfg', None) or {}",
        "_dt_cfg = _dt_cfg_raw or {}", 1)
    src = src.replace(
        "_gd_cfg = getattr(self, 'gt_dup_cfg', None) or {}",
        "_gd_cfg = _gd_cfg_raw or {}", 1)
    print('HEAD: fixed kwargs pop for D1+D3')

open(HEAD, 'w', encoding='utf-8').write(src)
py_compile.compile(HEAD, doraise=True)
print('HEAD: SYNTAX OK')

# Fix D2: InstanceData attribute issue
DET = '/home/yelingfei/projects/GCD/mmdet/models/detectors/gdino_inc_gcd.py'
src2 = open(DET, encoding='utf-8').read()

# Replace stash method to not set attributes on InstanceData
old_stash = '''    def _sel_grmi_stash_gt(self, batch_data_samples):
        if not self._sel_grmi_enable: return
        gt_list = []
        for ds in batch_data_samples:
            gt = ds.gt_instances
            gt._sel_meta = {'img_shape': ds.metainfo.get('img_shape', (800, 800))}
            gt_list.append(gt)
        self._sel_grmi_batch_gt = gt_list'''

new_stash = '''    def _sel_grmi_stash_gt(self, batch_data_samples):
        if not self._sel_grmi_enable: return
        gt_list = []
        self._sel_grmi_img_shapes = []
        for ds in batch_data_samples:
            gt_list.append(ds.gt_instances)
            self._sel_grmi_img_shapes.append(ds.metainfo.get('img_shape', (800, 800)))
        self._sel_grmi_batch_gt = gt_list'''

if old_stash in src2:
    src2 = src2.replace(old_stash, new_stash, 1)
    print('DET: fixed stash method')

# Fix the read side in _sel_grmi_apply
old_read = '''            meta = getattr(gt_inst, '_sel_meta', {})
            ih, iw = meta.get('img_shape', (800, 800))'''
new_read = '''            ih, iw = self._sel_grmi_img_shapes[bi] if bi < len(getattr(self, '_sel_grmi_img_shapes', [])) else (800, 800)'''

if old_read in src2:
    src2 = src2.replace(old_read, new_read, 1)
    print('DET: fixed meta read')

open(DET, 'w', encoding='utf-8').write(src2)
py_compile.compile(DET, doraise=True)
print('DET: SYNTAX OK')
