import numpy as np
from orbit_calc._sgp4_binding import LockFreePerturbationEngine

print("=" * 70)
print("  Detailed Perturbation Engine Debug")
print("=" * 70)

eng = LockFreePerturbationEngine()

print("\n[1] Testing different altitude density calculations:")
print(f"{'Altitude':>10}  {'Before update':>20}  {'After storm update':>20}")
print("-" * 70)

for alt in [200, 300, 350, 400, 421.9, 450, 500, 600, 800]:
    x, y, z = 6378.137 + alt, 0.0, 0.0
    vx, vy, vz = 0.0, 7.6, 0.0
    
    params_before = eng.get_all()
    eng.apply_perturbation(x, y, z, vx, vy, vz, 2460000.0, 60.0)
    dv_before = (vy - 7.6) * 1000
    
    eng.update_param(0, 250.0)
    eng.update_param(1, 200.0)
    eng.update_param(2, 8.5)
    eng.update_param(8, 5.0)
    
    x2, y2, z2 = 6378.137 + alt, 0.0, 0.0
    vx2, vy2, vz2 = 0.0, 7.6, 0.0
    eng.apply_perturbation(x2, y2, z2, vx2, vy2, vz2, 2460000.0, 60.0)
    dv_after = (vy2 - 7.6) * 1000
    
    eng.update_param(0, 150.0)
    eng.update_param(1, 150.0)
    eng.update_param(2, 2.0)
    eng.update_param(8, 0.0)
    
    print(f"{alt:>9} km  {dv_before:>18.6f} mm/s  {dv_after:>18.6f} mm/s  ratio: {dv_after/max(dv_before,1e-12):>6.1f}x")

print("\n[2] Testing parameter update and get:")
eng.update_param(0, 200.0)
eng.update_param(1, 180.0)
eng.update_param(2, 6.0)
eng.update_param(5, 2.2)
eng.update_param(6, 1.3)
eng.update_param(7, 0.02)
eng.update_param(8, 4.0)

params = eng.get_all()
print(f"  Params after update:")
for i, name in enumerate(['f10_7', 'f10_7_avg', 'kp', 'kp_3h', 'rho_set', 'Cd', 'Cr', 'Am', 'storm', 'seq']):
    print(f"    {name:>10} = {params[i]}")

print("\n[3] Testing with very high density to ensure calculation works:")
eng.update_param(4, 1e-8)

x, y, z = 6700.0, 0.0, 0.0
vx, vy, vz = 0.0, 7.8, 0.0
print(f"  Before: v={vy:.9f} km/s")
eng.apply_perturbation(x, y, z, vx, vy, vz, 2460000.0, 60.0)
print(f"  After:  v={vy:.9f} km/s")
print(f"  Delta v = {(vy - 7.8)*1e6:.3f} μm/s")

if abs(vy - 7.8) > 1e-12:
    print("\n  [OK] Perturbation is working with direct rho setting!")
else:
    print("\n  [FAIL] Perturbation still produces no change!")

print("\n[4] Testing batch perturbation:")
eng.update_param(0, 200.0)
eng.update_param(1, 180.0)
eng.update_param(2, 6.0)
eng.update_param(8, 3.0)
eng.update_param(7, 0.02)

n = 10
r = np.zeros((n, 3), dtype=np.float64)
v = np.zeros((n, 3), dtype=np.float64)
jd = np.zeros(n, dtype=np.float64)

for i in range(n):
    alt = 400 + i * 5
    r[i] = [6378.137 + alt, 0.0, 0.0]
    v[i] = [0.0, 7.6, 0.0]
    jd[i] = 2460000.0

r_before = r.copy()
v_before = v.copy()

eng.apply_perturbation_batch(r[:, 0], r[:, 1], r[:, 2], v[:, 0], v[:, 1], v[:, 2], jd, 60.0)

print(f"  {'Alt':>8}  {'dv (mm/s)':>12}")
print("  " + "-" * 25)
for i in range(n):
    alt = 400 + i * 5
    dv = (v[i, 1] - v_before[i, 1]) * 1000
    print(f"  {alt:>7}  {dv:>12.6f}")

print("\n" + "=" * 70)
