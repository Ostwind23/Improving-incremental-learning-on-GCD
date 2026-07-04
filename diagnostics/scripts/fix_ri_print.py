with open('/root/autodl-tmp/gcd_work/GCD/mmdet/models/detectors/gdino_inc_gcd.py') as f:
    code = f.read()
old = "_ri.net[0].out_features"
new = "_ri.W_m.out_features if hasattr(_ri, 'W_m') else _ri.net[0].out_features"
code = code.replace(old, new)
with open('/root/autodl-tmp/gcd_work/GCD/mmdet/models/detectors/gdino_inc_gcd.py', 'w') as f:
    f.write(code)
print('FIXED')
