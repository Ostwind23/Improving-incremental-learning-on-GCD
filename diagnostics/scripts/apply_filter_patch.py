"""Apply filter_old_pseudo_grad edits to gdino_head_inc_gcd.py."""
import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

# Edit 1: Insert filter_old_pseudo_grad after ap_rescue_v1_cfg line
for i, line in enumerate(lines):
    if 'self.ap_rescue_v1_cfg = kwargs.pop' in line:
        lines.insert(i + 1, "        self.filter_old_pseudo_grad = kwargs.pop('filter_old_pseudo_grad', False)\n")
        print(f'Edit 1 OK: line {i+2}')
        break
else:
    print('Edit 1 FAILED: ap_rescue_v1_cfg not found')
    sys.exit(1)

# Edit 2: Insert loss_by_feat_single override before _gt_dup_augment
for i, line in enumerate(lines):
    if 'def _gt_dup_augment(self, batch_gt_instances):' in line:
        override = [
            '\n',
            '    def loss_by_feat_single(self, cls_scores, bbox_preds,\n',
            '                            batch_gt_instances, batch_img_metas):\n',
            '        """Override: filter old-class pseudo-label gradient from R(M)."""\n',
            '        if not self.filter_old_pseudo_grad:\n',
            '            return super().loss_by_feat_single(\n',
            '                cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)\n',
            '\n',
            '        orig_get_targets = self.get_targets\n',
            '        NEW_TOKEN_START = 169\n',
            '\n',
            '        def filtered_get_targets(cls_scores_list, bbox_preds_list, bgti, bmi):\n',
            '            result = orig_get_targets(cls_scores_list, bbox_preds_list, bgti, bmi)\n',
            '            labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, \\\n',
            '                num_total_pos, num_total_neg = result\n',
            '\n',
            '            n_removed = 0\n',
            '            for i in range(len(labels_list)):\n',
            '                lab = labels_list[i]\n',
            '                matched = lab.sum(-1) > 0\n',
            '                if matched.sum() == 0:\n',
            '                    continue\n',
            '                for qi in range(lab.shape[0]):\n',
            '                    if not matched[qi]:\n',
            '                        continue\n',
            '                    hot = lab[qi].nonzero(as_tuple=True)[0]\n',
            '                    if len(hot) > 0 and (hot < NEW_TOKEN_START).any():\n',
            '                        label_weights_list[i][qi] = 0.0\n',
            '                        bbox_weights_list[i][qi, :] = 0.0\n',
            '                        n_removed += 1\n',
            '\n',
            '            if isinstance(num_total_pos, torch.Tensor):\n',
            '                num_total_pos = max(int(num_total_pos.item()) - n_removed, 1)\n',
            '            else:\n',
            '                num_total_pos = max(num_total_pos - n_removed, 1)\n',
            '\n',
            '            return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,\n',
            '                    num_total_pos, num_total_neg)\n',
            '\n',
            '        self.get_targets = filtered_get_targets\n',
            '        try:\n',
            '            return super().loss_by_feat_single(\n',
            '                cls_scores, bbox_preds, batch_gt_instances, batch_img_metas)\n',
            '        finally:\n',
            '            self.get_targets = orig_get_targets\n',
        ]
        for j, code_line in enumerate(override):
            lines.insert(i + j, code_line)
        print(f'Edit 2 OK: {len(override)} lines before line {i+1}')
        break
else:
    print('Edit 2 FAILED: _gt_dup_augment not found')
    sys.exit(1)

with open(sys.argv[1], 'w') as f:
    f.writelines(lines)
print(f'Done. Total lines: {len(lines)}')
