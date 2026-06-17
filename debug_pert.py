import numpy as np
from orbit_calc._sgp4_binding import LockFreePerturbationEngine

eng = LockFreePerturbationEngine()

print("=== Debugging LockFreePerturbationEngine ===")
print(f"PARAM_COUNT =", eng.PARAM_COUNT)

params = eng.get_all()
print("\nInitial params:", params)

eng.update_param(0, 200.0)
eng.update_param(1, 200.0)
eng.update_param(2, 7.5)
eng.update_param(5, 2.2)
eng.update_param(6, 1.0)
eng.update_param(7, 0.02)
eng.update_param(8, 4.0)

params2 = eng.get_all()
print("\nAfter update:")
print("  f10_7 =", params2[0])
print("  f10_7_avg =", params2[1])
print("  kp =", params2[2])
print("  Cd =", params2[5])
print("  Cr =", params2[6])
print("  Am =", params2[7])
print("  storm =", params2[8])
print("  seq =", params2[9])

print("\n=== Testing single point perturbation ===")
x, y, z = 6800.0, 0.0, 0.0
vx, vy, vz = 0.0, 7.6, 0.0
jd = 2460000.0
dt = 60.0

r = np.sqrt(x*x + y*y + z*z)
altitude = r - 6378.137
print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}) km")
print(f"Velocity: ({vx:.6f}, {vy:.6f}, {vz:.6f}) km/s")
print(f"Altitude: {altitude:.1f} km")
print(f"|v|: {np.sqrt(vx*vx + vy*vy + vz*vz):.3f} km/s")

print("\nCalling apply_perturbation...")
eng.apply_perturbation(x, y, z, vx, vy, vz, jd, dt)

print(f"\nAfter perturbation:")
print(f"Position: ({x:.6f}, {y:.6f}, {z:.6f}) km")
print(f"Velocity: ({vx:.9f}, {vy:.9f}, {vz:.9f}) km/s")

dx = x - 6800.0
dv = vy - 7.6
print(f"\nDeltas: dr={dx*1000:.6f} mm, dv={dv*1e6:.6f} μm/s")

print("\n=== Manual density calculation ===")
rho_storm = 6.0e-12 * 10
Cd = 2.2
Am = 0.02
v = 7600.0
f_drag = 0.5 * Cd * Am * rho_storm * v
a = f_drag
dv_expected = a * dt
print(f"rho = {rho_storm:.3e} kg/m^3")
print(f"Cd = {Cd}")
print(f"Am = {Am} m^2/kg")
print(f"v = {v} m/s")
print(f"f_drag = {f_drag:.3e} N")
print(f"a = {a:.3e} m/s^2")
print(f"Expected dv in {dt}s = {dv_expected*1000:.6f} mm/s")
print(f"Expected orbit decay in 1 day = {dv_expected * 86400 * 1000:.3f} m")
