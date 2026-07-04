import torch
ckpt = torch.load('/home/yelingfei/projects/GCD/work_dirs/grmi_rms_gate_3e_v2/epoch_3.pth', map_location='cpu')
for k, v in ckpt['state_dict'].items():
    if 'gate_net' in k:
        print(f'{k}: shape={v.shape}, norm={v.norm():.4f}, mean={v.mean():.6f}')
