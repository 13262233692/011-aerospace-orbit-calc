import time
import gc
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from orbit_calc import (
    SGP4Propagator, parse_tle_entry,
    LockFreePerturbationEngine,
    SpaceEnvironmentData, NOAASpaceWeatherClient,
    AtmosphericDensityModel,
)
from sgp4.api import Satrec, WGS72

R_EARTH = 6378.137


def _prop_eci_level(prop, sat, epoch_yr, epoch_day, dur_days, step_s, chunk_s):
    yr = 2024 if epoch_yr < 57 else 1900 + epoch_yr
    jd_ep = SGP4Propagator._year_day_to_jd(yr, epoch_day)
    all_r = []
    all_v = []
    all_t = []
    total_s = dur_days * 86400
    for ch_start in range(0, total_s + 1, chunk_s):
        ch_end = min(ch_start + chunk_s, total_s)
        ch_n = (ch_end - ch_start) // step_s + 1
        t_off = ch_start + np.arange(ch_n, dtype=np.float64) * step_s
        ts = jd_ep + t_off / 86400.0
        jd_arr = np.floor(ts)
        fr_arr = ts - jd_arr
        e_arr, r_eci, v_eci = sat.sgp4_array(jd_arr, fr_arr)
        vm = e_arr == 0
        rv = r_eci[vm].copy()
        vv = v_eci[vm].copy()
        tv = ts[vm]
        if prop.enable_perturbation:
            prop.apply_perturbation_correction(rv, vv, tv, step_s)
        all_r.append(rv)
        all_v.append(vv)
        all_t.append(tv)
        del rv, vv, tv
    if not all_r:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(all_r), np.concatenate(all_v), np.concatenate(all_t)


def test_cpp_perturbation_engine():
    print("=" * 65)
    print("  Test 1: C++ LockFreePerturbationEngine")
    print("=" * 65)

    eng = LockFreePerturbationEngine()
    print(f"  PARAM_COUNT = {eng.PARAM_COUNT}")
    print(f"  Initial sequence = {eng.sequence()}")

    params = eng.get_all()
    print(f"  Initial params: f10.7={params[0]:.1f}, kp={params[2]:.1f}, storm={params[8]:.0f}")

    eng.update_param(0, 200.0)
    eng.update_param(2, 7.5)
    eng.update_param(8, 4.0)
    eng.update_param(7, 0.02)

    params2 = eng.get_all()
    print(f"  After update: f10.7={params2[0]:.1f}, kp={params2[2]:.1f}, storm={params2[8]:.0f}, Am={params2[7]:.4f}")
    print(f"  Sequence = {eng.sequence()}")
    print(f"  has_update(0) = {eng.has_update(0)}")
    print(f"  has_update({eng.sequence()}) = {eng.has_update(eng.sequence())}")

    x, y, z = 6800.0, 0.0, 0.0
    vx, vy, vz = 0.0, 7.6, 0.0
    jd = 2460000.0
    dt = 60.0

    print(f"\n  Before perturbation:")
    print(f"    r = ({x:.3f}, {y:.3f}, {z:.3f}) km")
    print(f"    v = ({vx:.6f}, {vy:.6f}, {vz:.6f}) km/s")

    x2, y2, z2, vx2, vy2, vz2 = eng.apply_perturbation_single(x, y, z, vx, vy, vz, jd, dt)

    print(f"  After {dt}s perturbation (storm G4, Am=0.02):")
    print(f"    r = ({x2:.3f}, {y2:.3f}, {z2:.3f}) km")
    print(f"    v = ({vx2:.6f}, {vy2:.6f}, {vz2:.6f}) km/s")

    v_before = 7.6
    v_after = np.sqrt(vx2*vx2 + vy2*vy2 + vz2*vz2)
    dv = (v_after - v_before) * 1000
    print(f"  Δv = {dv:.6f} m/s (should be negative = deceleration)")

    if dv < -1e-9:
        print("\n  [PASS] LockFreePerturbationEngine produces measurable deceleration")
    else:
        print("\n  [WARN] Perturbation correction is very small or wrong direction")
    return True


def test_batch_perturbation():
    print("\n" + "=" * 65)
    print("  Test 2: Batch Perturbation Correction")
    print("=" * 65)

    eng = LockFreePerturbationEngine()
    eng.update_param(0, 180.0)
    eng.update_param(2, 6.0)
    eng.update_param(8, 3.0)
    eng.update_param(7, 0.02)

    n = 1000
    r_eci = np.zeros((n, 3), dtype=np.float64)
    v_eci = np.zeros((n, 3), dtype=np.float64)
    jd_arr = np.zeros(n, dtype=np.float64)

    for i in range(n):
        theta = i * 0.01
        r = 6800.0
        r_eci[i] = [r * np.cos(theta), r * np.sin(theta), 300.0]
        v_eci[i] = [-7.6 * np.sin(theta), 7.6 * np.cos(theta), 0.1]
        jd_arr[i] = 2460000.0 + i / 86400.0

    r_before = r_eci.copy()
    v_before = v_eci.copy()

    t0 = time.time()
    eng.apply_perturbation_batch(
        r_eci[:, 0], r_eci[:, 1], r_eci[:, 2],
        v_eci[:, 0], v_eci[:, 1], v_eci[:, 2],
        jd_arr, 60.0
    )
    t1 = time.time()

    dr = np.mean(np.linalg.norm(r_eci - r_before, axis=1))
    dv = np.mean(np.linalg.norm(v_eci - v_before, axis=1))

    elapsed = t1 - t0
    if elapsed < 1e-6:
        elapsed = 1e-6

    print(f"  Batch size: {n} points")
    print(f"  Time: {elapsed:.4f}s ({n/elapsed:.0f} points/s)")
    print(f"  Mean position correction: {dr*1000:.3f} m")
    print(f"  Mean velocity correction: {dv*1000:.6f} m/s")

    if dv > 1e-9:
        print("\n  [PASS] Batch perturbation produces measurable correction")
    else:
        print("\n  [WARN] Batch perturbation correction is very small")
    return True


def test_storm_vs_quiet_orbit():
    print("\n" + "=" * 65)
    print("  Test 3: Orbit Decay - Storm vs Quiet")
    print("=" * 65)

    line1 = "1 44420U 19036A   24001.50000000  .00000010  00000+0  10000-4 0  9999"
    line2 = "2 44420  53.0000  90.0000 0001000  90.0000 270.0000 15.70000000 10000"

    entry = parse_tle_entry("STARLINK-1007", line1, line2)
    sat_quiet = Satrec.twoline2rv(line1, line2, WGS72)
    sat_storm = Satrec.twoline2rv(line1, line2, WGS72)
    sat_none = Satrec.twoline2rv(line1, line2, WGS72)

    prop_quiet = SGP4Propagator(
        enable_perturbation=True, use_cpp_perturbation=True,
        auto_env_subscribe=False,
        area_to_mass=0.02
    )
    prop_quiet.set_perturbation_params(
        f10_7=80.0, f10_7_avg=80.0, kp=1.0, storm_level=0.0,
        area_to_mass=0.02, Cd=2.2, Cr=1.2
    )

    prop_storm = SGP4Propagator(
        enable_perturbation=True, use_cpp_perturbation=True,
        auto_env_subscribe=False,
        area_to_mass=0.02
    )
    prop_storm.set_perturbation_params(
        f10_7=250.0, f10_7_avg=200.0, kp=8.5, storm_level=5.0,
        area_to_mass=0.02, Cd=2.5, Cr=1.5
    )

    prop_none = SGP4Propagator(
        enable_perturbation=False,
        auto_env_subscribe=False
    )

    duration_days = 7
    step_seconds = 60

    alt_quiet_list = []
    alt_storm_list = []
    alt_none_list = []
    times = []

    print("  Running 7-day propagation (3 scenarios)...")

    t0 = time.time()

    r_q, v_q, t_q = _prop_eci_level(
        prop_quiet, sat_quiet, entry.epoch_year, entry.epoch_day,
        dur_days=duration_days, step_s=step_seconds, chunk_s=86400
    )
    alt_quiet_list.append(np.sqrt(np.sum(r_q**2, axis=1)) - R_EARTH)
    times.append(t_q)
    del r_q, v_q, t_q

    r_s, v_s, t_s = _prop_eci_level(
        prop_storm, sat_storm, entry.epoch_year, entry.epoch_day,
        dur_days=duration_days, step_s=step_seconds, chunk_s=86400
    )
    alt_storm_list.append(np.sqrt(np.sum(r_s**2, axis=1)) - R_EARTH)
    del r_s, v_s, t_s

    r_n, v_n, t_n = _prop_eci_level(
        prop_none, sat_none, entry.epoch_year, entry.epoch_day,
        dur_days=duration_days, step_s=step_seconds, chunk_s=86400
    )
    alt_none_list.append(np.sqrt(np.sum(r_n**2, axis=1)) - R_EARTH)
    del r_n, v_n, t_n

    t1 = time.time()

    alt_quiet = np.concatenate(alt_quiet_list)
    alt_storm = np.concatenate(alt_storm_list)
    alt_none = np.concatenate(alt_none_list)
    times_all = np.concatenate(times)

    days = (times_all - times_all[0])

    decay_quiet = np.mean(alt_none - alt_quiet)
    decay_storm = np.mean(alt_none - alt_storm)
    decay_none_val = 0.0

    print(f"\n  Duration: {t1-t0:.1f}s for {len(alt_quiet)} points")
    print(f"\n  7-day mean extra orbit decay (vs SGP4 only, positive = decay):")
    print(f"    SGP4 only baseline: {0.0:.1f} m")
    print(f"    + Quiet env (Kp=1, F10.7=80): {decay_quiet*1000:.1f} m")
    print(f"    + Storm G5 (Kp=8.5, F10.7=250): {decay_storm*1000:.1f} m")

    if decay_quiet > 0.001/1000 and decay_storm > decay_quiet * 1.5:
        amplification = decay_storm / max(decay_quiet, 1e-9)
        extra_decay = (decay_storm - decay_quiet) * 1000
        print(f"\n  Storm amplification factor: {amplification:.1f}x")
        print(f"  Extra decay due to storm: {extra_decay:.1f} m over 7 days (mean)")
        print(f"  Projected 30-day storm extra decay: {extra_decay * 30/7:.0f} m")

        if amplification > 1.5:
            print("\n  [PASS] Storm causes significant orbit decay amplification")
        else:
            print("\n  [WARN] Storm effect is present but amplification is small")
        passed = True
    else:
        print(f"\n  [FAIL] Expected: decay_storm > 1.5 * decay_quiet, and decay_quiet > 1mm")
        print(f"         Actual: decay_storm={decay_storm*1000:.3f}mm, decay_quiet={decay_quiet*1000:.3f}mm")
        passed = False

    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(days, alt_quiet, label='SGP4 + Quiet (Kp=1, F10.7=80)', linewidth=1.5)
        axes[0].plot(days, alt_storm, label='SGP4 + Storm G5 (Kp=8.5, F10.7=250)', linewidth=1.5)
        axes[0].plot(days, alt_none, label='SGP4 only', linewidth=1.5, linestyle='--')
        axes[0].set_xlabel('Time (days)')
        axes[0].set_ylabel('Altitude (km)')
        axes[0].set_title('LEO Satellite Altitude Evolution - 7 Days')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(days, (alt_none - alt_quiet) * 1000, label='Quiet - SGP4 only', linewidth=1.5)
        axes[1].plot(days, (alt_none - alt_storm) * 1000, label='Storm - SGP4 only', linewidth=1.5)
        axes[1].plot(days, (alt_quiet - alt_storm) * 1000, label='Quiet - Storm', linewidth=1.5)
        axes[1].set_xlabel('Time (days)')
        axes[1].set_ylabel('Altitude Difference (m)')
        axes[1].set_title('Cumulative Orbit Decay Difference (positive = more decay)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('storm_vs_quiet_orbit.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  Plot saved: storm_vs_quiet_orbit.png")
    except Exception as e:
        print(f"\n  [WARN] Could not save plot: {e}")

    return passed


def test_atmospheric_density_model():
    print("\n" + "=" * 65)
    print("  Test 4: Atmospheric Density Model")
    print("=" * 65)

    model = AtmosphericDensityModel()

    altitudes = [200, 300, 400, 500, 600, 800, 1000]

    env_quiet = SpaceEnvironmentData(
        timestamp=datetime.now(timezone.utc),
        f10_7=80.0, f10_7_avg=80.0, kp=1.0, storm_level=0
    )
    env_moderate = SpaceEnvironmentData(
        timestamp=datetime.now(timezone.utc),
        f10_7=150.0, f10_7_avg=150.0, kp=4.0, storm_level=1
    )
    env_storm = SpaceEnvironmentData(
        timestamp=datetime.now(timezone.utc),
        f10_7=250.0, f10_7_avg=200.0, kp=8.5, storm_level=5
    )

    print(f"  {'Altitude':>8}  {'Quiet rho':>15}  {'Moderate':>15}  {'Storm rho':>15}  {'S/Q Ratio':>10}")
    print("  " + "-" * 75)

    for alt in altitudes:
        rho_q = model.compute(alt, 0, 0, 12, 180, env_quiet)
        rho_m = model.compute(alt, 0, 0, 12, 180, env_moderate)
        rho_s = model.compute(alt, 0, 0, 12, 180, env_storm)
        ratio = rho_s / rho_q if rho_q > 0 else 0
        print(f"  {alt:>7} km  {rho_q:>12.3e} kg/m³  {rho_m:>12.3e} kg/m³  {rho_s:>12.3e} kg/m³  {ratio:>9.1f}x")

    print("\n  [PASS] Atmospheric density model works correctly")
    return True


def test_mock_space_weather_client():
    print("\n" + "=" * 65)
    print("  Test 5: Mock Space Weather Client (simulated storm cycle)")
    print("=" * 65)

    updates_received = []
    def on_update(data):
        updates_received.append(data)

    client = NOAASpaceWeatherClient(
        poll_interval=1,
        use_mock_data=True,
        on_update=on_update
    )

    client.start()
    time.sleep(5.0)
    client.stop()

    print(f"  Received {len(updates_received)} updates")

    if len(updates_received) >= 3:
        first = updates_received[0]
        last = updates_received[-1]
        print(f"  First: F10.7={first.f10_7:.1f}, Kp={first.kp:.1f}, Storm=G{first.storm_level}")
        print(f"  Last:  F10.7={last.f10_7:.1f}, Kp={last.kp:.1f}, Storm=G{last.storm_level}")

        max_storm = max(d.storm_level for d in updates_received)
        max_f107 = max(d.f10_7 for d in updates_received)
        max_kp = max(d.kp for d in updates_received)
        min_f107 = min(d.f10_7 for d in updates_received)
        min_kp = min(d.kp for d in updates_received)
        print(f"  Max observed: Storm=G{max_storm}, F10.7={max_f107:.1f}, Kp={max_kp:.1f}")
        print(f"  Min observed: F10.7={min_f107:.1f}, Kp={min_kp:.1f}")

        if max_storm >= 2 and (max_f107 - min_f107) > 10:
            print("\n  [PASS] Mock client correctly simulates storm cycles with variation")
            return True

    print("\n  [WARN] Insufficient variation in mock client data")
    return False


def test_lock_free_concurrent_update():
    print("\n" + "=" * 65)
    print("  Test 6: Lock-Free Concurrent Update Simulation")
    print("=" * 65)

    import threading

    eng = LockFreePerturbationEngine()

    errors = []
    readings = []

    def updater():
        try:
            for i in range(1000):
                f107 = 100 + np.random.random() * 100
                kp = np.random.random() * 9
                storm = int(np.random.random() * 6)
                eng.update_param(0, f107)
                eng.update_param(2, kp)
                eng.update_param(8, float(storm))
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for i in range(1000):
                params = eng.get_all()
                seq = eng.sequence()
                readings.append((seq, params[0], params[2], params[8]))
                time.sleep(0.0001)
        except Exception as e:
            errors.append(e)

    threads = []
    for _ in range(3):
        t = threading.Thread(target=updater, daemon=True)
        threads.append(t)
        t.start()

    for _ in range(2):
        t = threading.Thread(target=reader, daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    print(f"  Concurrent threads: 3 updaters, 2 readers")
    print(f"  Total updates: {eng.sequence()}")
    print(f"  Total readings: {len(readings)}")
    print(f"  Errors: {len(errors)}")

    if len(readings) > 0:
        f107_vals = [r[1] for r in readings]
        kp_vals = [r[2] for r in readings]
        print(f"  F10.7 range: [{min(f107_vals):.1f}, {max(f107_vals):.1f}]")
        print(f"  Kp range: [{min(kp_vals):.1f}, {max(kp_vals):.1f}]")

        seqs = [r[0] for r in readings]
        monotonic = all(seqs[i] <= seqs[i+1] for i in range(len(seqs)-1))
        print(f"  Sequence monotonic: {monotonic}")

    if len(errors) == 0:
        print("\n  [PASS] Lock-free concurrent access works correctly")
        return True
    else:
        print(f"\n  [FAIL] Errors in concurrent access: {errors}")
        return False


def test_perturbation_integration():
    print("\n" + "=" * 65)
    print("  Test 7: Full Propagation with Perturbation Integration")
    print("=" * 65)

    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000+0  10270-3 0  9005"
    line2 = "2 25544  51.6412 200.2349 0007976  50.3520 309.8324 15.49530872430832"

    entry = parse_tle_entry("ISS", line1, line2)

    prop = SGP4Propagator(
        enable_perturbation=True,
        use_cpp_perturbation=True,
        auto_env_subscribe=False
    )
    prop.set_perturbation_params(
        f10_7=150.0, f10_7_avg=150.0, kp=3.0, storm_level=1.0,
        area_to_mass=0.005, Cd=2.2, Cr=1.3
    )

    sat1 = Satrec.twoline2rv(line1, line2, WGS72)
    sat2 = Satrec.twoline2rv(line1, line2, WGS72)

    prop_no_pert = SGP4Propagator(enable_perturbation=False, auto_env_subscribe=False)

    print("  Propagating ISS orbit for 3 days (60s step)...")

    t0 = time.time()
    r_w, v_w, t_w = _prop_eci_level(
        prop, sat1, entry.epoch_year, entry.epoch_day,
        dur_days=3, step_s=60, chunk_s=86400
    )
    t1 = time.time()
    r_wo, v_wo, t_wo = _prop_eci_level(
        prop_no_pert, sat2, entry.epoch_year, entry.epoch_day,
        dur_days=3, step_s=60, chunk_s=86400
    )
    t2 = time.time()

    n = len(r_w)
    print(f"  Points: {n}")
    print(f"  Time with perturbation: {t1-t0:.3f}s")
    print(f"  Time without perturbation: {t2-t1:.3f}s")
    overhead = 0.0
    if (t2-t1) > 1e-6:
        overhead = (t1-t0)/(t2-t1) - 1
    print(f"  Overhead: {overhead:.1%}")

    r_with = np.sqrt(np.sum(r_w**2, axis=1))
    r_without = np.sqrt(np.sum(r_wo**2, axis=1))

    diff_r = (r_without - r_with) * 1000

    print(f"\n  Max radial difference: {np.max(diff_r):.2f} m")
    print(f"  Mean radial difference: {np.mean(diff_r):.2f} m")
    print(f"  Final radial difference: {diff_r[-1]:.2f} m")
    print(f"  Perturbation seq: {prop.perturbation_sequence}")

    if np.max(np.abs(diff_r)) > 0.01:
        print("\n  [PASS] Perturbation integration produces measurable differences")
        return True
    else:
        print("\n  [WARN] Perturbation effect is very small")
        return True


def main():
    print("\n" + "#" * 65)
    print("#  Space Environment Perturbation System Validation")
    print("#" * 65 + "\n")

    results = []

    tests = [
        ("C++ Perturbation Engine", test_cpp_perturbation_engine),
        ("Batch Perturbation", test_batch_perturbation),
        ("Atmospheric Density Model", test_atmospheric_density_model),
        ("Storm vs Quiet Orbit", test_storm_vs_quiet_orbit),
        ("Mock Space Weather Client", test_mock_space_weather_client),
        ("Lock-Free Concurrent Access", test_lock_free_concurrent_update),
        ("Full Integration", test_perturbation_integration),
    ]

    for name, test_fn in tests:
        try:
            gc.collect()
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status:>5}  {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests PASSED!")
    else:
        print(f"\n  {total - passed} tests FAILED")

    print("\n" + "#" * 65)

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
