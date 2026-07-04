import torch, numpy as np
ckpt = torch.load('work_dirs/grmi_bilinear_12e_clean/epoch_12.pth', map_location='cpu')
sd = ckpt['state_dict']
W_m = sd['residual_inject.W_m.weight']
W_t = sd['residual_inject.W_t.weight']
W_out = sd['residual_inject.W_out.weight']
print(f'LN-trained: ||W_m||={W_m.norm():.1f} ||W_t||={W_t.norm():.1f} ||W_out||={W_out.norm():.1f}')

torch.manual_seed(42)
N=500
res=[]
for _ in range(N):
    T = torch.randn(1,256); T = T/T.norm()*16
    M_new = T * 2 + torch.randn(1,256)*5
    M_old = torch.randn(1,256)*15
    M_bg = torch.randn(1,256)*15
    for label, M in [('new',M_new),('old',M_old),('bg',M_bg)]:
        h = (M @ W_m.T) * (T @ W_t.T)
        R = h @ W_out.T
        res.append({'label':label,'r_norm':R.norm().item()})

for l in ['new','old','bg']:
    v=[r['r_norm'] for r in res if r['label']==l]
    v=np.array(v)
    print(f'  {l}: |R|={v.mean():.0f} median={np.median(v):.0f} new/old ratio={v.mean()/max(v.mean(),1e-8):.2f}')
