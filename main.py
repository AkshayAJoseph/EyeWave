"""
main.py
=======
Entry point for EyeWave.

Runs the main capture + gaze-pipeline loop.  All business logic lives in
src/ — this file only wires the components together and handles key events.

Usage
-----
  python main.py

Keyboard shortcuts
------------------
  C          — Calibrate eye spheres  (loads saved calibration if available)
  1 2 3 4   — Start 4-point corner calibration (TL / TR / BL / BR)
  SPACE      — Confirm current calibration corner
  TAB        — Switch layout  (QWERTY ↔ AAC)
  M          — Switch selection mode  (Scanning ↔ Gaze-dwell)
  B          — Toggle blink selection on / off
  X          — Drop a debug gaze marker on the monitor plane
  F7         — Toggle OS mouse cursor control
  J / L      — Orbit debug camera: yaw left / right
  I / K      — Orbit debug camera: pitch up / down
  [ / ]      — Orbit zoom out / in
  R          — Reset orbit camera
  Q          — Quit
"""

import math
import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    import pyautogui
    PYAUTOGUI_OK = True
    MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
except ImportError:
    PYAUTOGUI_OK = False
    MONITOR_WIDTH, MONITOR_HEIGHT = 1920, 1080

try:
    import keyboard as kb
    KB_OK = True
except ImportError:
    KB_OK = False

import mediapipe as mp
from scipy.spatial.transform import Rotation as Rscipy

# ── MediaPipe compatibility (old solutions API vs new Tasks API) ──────────
_USE_LEGACY_MP = False
try:
    _mp_fm = mp.solutions.face_mesh
    # Test-instantiate to catch protobuf 5.x runtime breakage
    _test = _mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1)
    _test.close()
    del _test
    _USE_LEGACY_MP = True
except Exception:
    from mediapipe.tasks.python import vision as _mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOpts

from src.config import (
    BASE_RADIUS,
    ORBIT_YAW, ORBIT_PITCH, ORBIT_RADIUS,
    NOSE_IDX, BLINK_EAR_THRESH,
    CALIB_FILE, GAZE_DATA_FILE, FACE_LANDMARKER,
)
from src.utils import (
    normalize, compute_scale, pca_orientation,
    create_monitor_plane, ray_plane_ab,
    CalibrationManager, GazeDataCollector,
)
from src.visionc import (
    AdaptiveGazeFilter, FixationDetector, SmartDwellController,
    MultiPointCalib, BlinkDetector, ScanningController,
    render_debug_view_orbit,
)
from src.interface import LayoutManager, EyeKeyboard


def main():
    # ── Orbit camera state (mutable in-place) ─────────────────────────────
    orbit = {
        'yaw':    ORBIT_YAW,
        'pitch':  ORBIT_PITCH,
        'radius': ORBIT_RADIUS,
        'frozen': False,
        'pivot':  None,
    }

    # ── MediaPipe ─────────────────────────────────────────────────────────
    if _USE_LEGACY_MP:
        face_mesh = _mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        _fl_opts = _mp_vision.FaceLandmarkerOptions(
            base_options=_BaseOpts(model_asset_path=FACE_LANDMARKER),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_mesh = _mp_vision.FaceLandmarker.create_from_options(_fl_opts)
        _mp_ts = 0   # monotonic frame timestamp for VIDEO mode
    cap = cv2.VideoCapture(0)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Pipeline objects ──────────────────────────────────────────────────
    layout       = LayoutManager()
    keyboard_gui = EyeKeyboard()
    calib        = MultiPointCalib()
    cal_mgr      = CalibrationManager()
    gf           = AdaptiveGazeFilter()
    fix_det      = FixationDetector()
    dwell        = SmartDwellController()
    scanner      = ScanningController()
    blinker      = BlinkDetector()
    collector    = GazeDataCollector()

    scanner.set_layout_size(layout.rows, layout.cols)

    # Selection mode: 'scan' = scanning primary, 'gaze' = dwell primary
    sel_mode = 'scan'

    # ── 3D tracker state ──────────────────────────────────────────────────
    R_ref   = [None]
    rdb     = deque(maxlen=6)       # tiny 3D buffer for debug view smoothing

    l_locked = r_locked = False
    l_off = r_off = l_sc = r_sc = None

    mon_corners = mon_cw = mon_nw = upc = None
    sw_l = sw_r = sr_l = sr_r = None
    i3l  = i3r  = None
    hc   = Rf   = npts = lms = None
    avg_dir     = None
    gaze_markers = []

    # ── Mouse thread ──────────────────────────────────────────────────────
    mouse_on  = False
    mt        = [MONITOR_WIDTH // 2, MONITOR_HEIGHT // 2]
    ml        = threading.Lock()

    def mouse_loop():
        while True:
            if mouse_on and PYAUTOGUI_OK:
                with ml:
                    x, y = mt
                pyautogui.moveTo(x, y)
            time.sleep(0.01)

    threading.Thread(target=mouse_loop, daemon=True).start()

    # ── Debounce timestamps ───────────────────────────────────────────────
    prev_f7    = False
    prev_space = False
    last_tab   = last_m = last_b = 0.0
    last_ft    = time.time()

    # ── Load saved calibration hint ───────────────────────────────────────
    if cal_mgr.exists():
        print(f"[EyeWave] Saved calibration found ({CALIB_FILE}).")
        print("  Press C to load it and set the monitor plane.")
    else:
        print("[EyeWave] No saved calibration — press C for fresh calibration.")

    # ── Fullscreen keyboard window ─────────────────────────────────────────
    cv2.namedWindow("EyeWave Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("EyeWave Keyboard",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    print("\nEyeWave  -  Eye-Tracking AAC Keyboard")
    print("-" * 45)
    print("  C        - Calibrate (loads saved if available)")
    print("  1-4+SPC  - 4-point corner calibration")
    print("  TAB      - Switch layout: QWERTY / AAC")
    print("  M        - Switch mode: Scanning / Gaze-dwell")
    print("  B        - Toggle blink selection")
    print("  Q        - Quit")
    print(f"  Gaze samples logged -> {GAZE_DATA_FILE}  ({collector.count} so far)")
    print()

    scanner.start()     # start scanning immediately on launch

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now  = time.time()
        dt   = min(now - last_ft, 0.10)    # cap at 100 ms to survive stalls
        last_ft = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks (legacy or Tasks API)
        if _USE_LEGACY_MP:
            results = face_mesh.process(frame_rgb)
            _face_found = bool(results.multi_face_landmarks)
            if _face_found:
                lms = results.multi_face_landmarks[0].landmark
        else:
            _mp_ts = int(time.time() * 1000)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = face_mesh.detect_for_video(mp_img, _mp_ts)
            _face_found = bool(results.face_landmarks)
            if _face_found:
                lms = results.face_landmarks[0]

        gaze_row_hint = gaze_col_hint = None   # hints for scanner

        # ── Blink update (ALWAYS runs, even when face lost) ───────────
        blinker.update(lms if _face_found else None, fw, fh)

        # ── Face mesh ─────────────────────────────────────────────────────
        if _face_found:
            # lms already set above by the detect block
            npts = np.array([[lms[i].x*fw, lms[i].y*fh, lms[i].z*fw]
                              for i in NOSE_IDX])
            hc, Rf = pca_orientation(npts, R_ref)

            li = lms[468]; ri = lms[473]
            i3l = np.array([li.x*fw, li.y*fh, li.z*fw])
            i3r = np.array([ri.x*fw, ri.y*fh, ri.z*fw])

            # Landmarks on cam preview
            for lm in lms:
                cv2.circle(frame, (int(lm.x*fw), int(lm.y*fh)),
                           0, (255, 255, 255), -1)

            # EAR value overlay on camera preview
            ear_col = (0, 0, 255) if blinker.ear < BLINK_EAR_THRESH else (0, 255, 0)
            cv2.putText(frame, f"EAR:{blinker.ear:.3f}", (fw - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_col, 2)

            # Eye sphere positions
            cns = compute_scale(npts)
            if l_locked and l_off is not None:
                sr = cns / l_sc if l_sc else 1.0
                sw_l = hc + Rf @ (l_off * sr);  sr_l = int(BASE_RADIUS * sr)
            if r_locked and r_off is not None:
                sr = cns / r_sc if r_sc else 1.0
                sw_r = hc + Rf @ (r_off * sr);  sr_r = int(BASE_RADIUS * sr)

            # Draw eye circles on cam preview
            if not l_locked:
                cv2.circle(frame, (int(li.x*fw), int(li.y*fh)),
                           10, (255, 25, 25), 2)
            elif sw_l is not None:
                cv2.circle(frame, (int(sw_l[0]), int(sw_l[1])),
                           sr_l, (255, 255, 25), 2)
            if not r_locked:
                cv2.circle(frame, (int(ri.x*fw), int(ri.y*fh)),
                           10, (25, 255, 25), 2)
            elif sw_r is not None:
                cv2.circle(frame, (int(sw_r[0]), int(sw_r[1])),
                           sr_r, (25, 255, 255), 2)

            # ── Gaze pipeline ──────────────────────────────────────────────
            if (l_locked and r_locked
                    and sw_l is not None and sw_r is not None):

                lg = normalize(i3l - sw_l)
                rg = normalize(i3r - sw_r)
                rd = normalize(lg + rg)
                rdb.append(rd)
                avg_dir = normalize(np.mean(rdb, axis=0))

                # Combined ray on cam preview
                Oc = ((sw_l + sw_r) * 0.5).astype(int)
                Tc = (Oc + avg_dir * 350).astype(int)
                cv2.line(frame, tuple(Oc[:2]), tuple(Tc[:2]), (255, 255, 10), 3)

                if mon_corners is not None:
                    O3  = (sw_l + sw_r) * 0.5
                    ab  = ray_plane_ab(O3, avg_dir,
                                       mon_corners, mon_cw, mon_nw)
                    if ab:
                        a_raw, b_raw = ab
                        a_raw = 1.0 - a_raw    # mirror horizontal
                        b_raw = 1.0 - b_raw    # mirror vertical

                        # Stage 1 — adaptive 2-D filter
                        a_f, b_f, speed = gf.update(a_raw, b_raw)

                        # Stage 2 — fixation detection
                        is_fix = fix_det.update(a_f, b_f, speed)

                        # Feed calibration (stable frames only)
                        if calib.active:
                            calib.record(a_raw, b_raw, fix_det.dispersion)

                        # Corrected fixation centroid
                        fa_c, fb_c = calib.correct(
                            fix_det.centroid_a, fix_det.centroid_b)

                        # Gaze hints for scanner
                        if is_fix:
                            gaze_row_hint = min(int(fb_c * layout.rows),
                                                layout.rows - 1)
                            gaze_col_hint = min(int(fa_c * layout.cols),
                                                layout.cols - 1)

                        # Stage 3 — dwell (gaze mode)
                        if sel_mode == 'gaze':
                            act = dwell.update(fa_c, fb_c, is_fix, dt,
                                               layout.rows, layout.cols)
                            if act:
                                result = keyboard_gui.activate_key(act, layout)
                                if result == '__SWAP__':
                                    layout.toggle()
                                    scanner.set_layout_size(
                                        layout.rows, layout.cols)
                                elif result and result not in ('__NUM__',):
                                    collector.log(i3l, i3r, Rf, a_raw, b_raw,
                                                  act[0], act[1], str(result))

                            # Blink → immediate activation of hovered key
                            if blinker.blink and dwell.hovered:
                                act2   = dwell.hovered
                                result = keyboard_gui.activate_key(act2, layout)
                                if result and result not in ('__SWAP__', '__NUM__'):
                                    collector.log(i3l, i3r, Rf, a_raw, b_raw,
                                                  act2[0], act2[1], str(result))

                        # OS mouse follows corrected smooth position
                        a_fc, b_fc = calib.correct(a_f, b_f)
                        sx = int(np.clip(a_fc, 0, 1) * MONITOR_WIDTH)
                        sy = int(np.clip(b_fc, 0, 1) * MONITOR_HEIGHT)
                        if mouse_on:
                            with ml:
                                mt[0] = sx;  mt[1] = sy

                        # Status label on cam preview
                        lbl = "FIX" if is_fix else f"sac{speed:.2f}"
                        cv2.putText(frame, lbl, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6,
                                    (0, 255, 120) if is_fix else (0, 120, 255), 1)

        # ── Scanning update (every frame, separate from gaze mode) ────────
        act_scan = scanner.update(blinker.blink, blinker.double_blink,
                                  gaze_row_hint, gaze_col_hint)
        if act_scan:
            result = keyboard_gui.activate_key(act_scan, layout)
            if result == '__SWAP__':
                layout.toggle()
                scanner.set_layout_size(layout.rows, layout.cols)
            elif result and i3l is not None and i3r is not None and Rf is not None:
                collector.log(i3l, i3r, Rf,
                              gf.a, gf.b,
                              act_scan[0], act_scan[1], str(result))

        # ── Render windows ─────────────────────────────────────────────────
        kbd = keyboard_gui.draw(layout, dwell, scanner, blinker,
                                fix_det, gf, calib, collector, sel_mode)
        cv2.imshow("EyeWave Keyboard", kbd)
        cv2.imshow("EyeWave Camera", frame)

        # ── Debug orbit view ───────────────────────────────────────────────
        lms3d = None
        if _face_found:
            lms3d = np.array([[p.x*fw, p.y*fh, p.z*fw]
                              for p in lms])
        render_debug_view_orbit(
            fh, fw,
            orbit_yaw=orbit['yaw'], orbit_pitch=orbit['pitch'],
            orbit_radius=orbit['radius'],
            debug_world_frozen=orbit['frozen'],
            orbit_pivot_frozen=orbit['pivot'],
            head_center3d=hc,
            sphere_world_l=sw_l if l_locked else None,
            scaled_radius_l=sr_l if l_locked else None,
            sphere_world_r=sw_r if r_locked else None,
            scaled_radius_r=sr_r if r_locked else None,
            iris3d_l=i3l, iris3d_r=i3r,
            left_locked=l_locked, right_locked=r_locked,
            landmarks3d=lms3d, combined_dir=avg_dir,
            gaze_len=5230, monitor_corners=mon_corners,
            monitor_center=mon_cw, monitor_normal=mon_nw,
            gaze_markers=gaze_markers, units_per_cm=upc,
        )

        # ── Orbit keyboard controls ────────────────────────────────────────
        if KB_OK:
            ys = math.radians(1.5);  ps = math.radians(1.5)
            if kb.is_pressed('j'): orbit['yaw']    -= ys
            if kb.is_pressed('l'): orbit['yaw']    += ys
            if kb.is_pressed('i'): orbit['pitch']  += ps
            if kb.is_pressed('k'): orbit['pitch']  -= ps
            if kb.is_pressed('['): orbit['radius'] += 12
            if kb.is_pressed(']'): orbit['radius']  = max(80., orbit['radius'] - 12)
            if kb.is_pressed('r'):
                orbit['yaw'] = 0.; orbit['pitch'] = 0.; orbit['radius'] = 600.
            orbit['pitch'] = max(math.radians(-89),
                                 min(math.radians(89), orbit['pitch']))
            f7n = kb.is_pressed('f7')
            if f7n and not prev_f7:
                mouse_on = not mouse_on
                print(f"[Mouse] {'ON' if mouse_on else 'OFF'}")
            prev_f7 = f7n

        # ── cv2.waitKey events ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # TAB — toggle layout
        elif key == 9 and now - last_tab > 0.5:
            layout.toggle()
            scanner.set_layout_size(layout.rows, layout.cols)
            scanner.start()
            keyboard_gui.status = f"Layout: {layout.current.upper()}"
            last_tab = now
            print(f"[Layout] → {layout.current.upper()}")

        # M — toggle selection mode
        elif key == ord('m') and now - last_m > 0.5:
            sel_mode = 'scan' if sel_mode == 'gaze' else 'gaze'
            if sel_mode == 'scan':
                scanner.start()
            else:
                scanner.stop()
            keyboard_gui.status = (
                "Mode: SCANNING (blink to select)"
                if sel_mode == 'scan'
                else "Mode: GAZE DWELL"
            )
            last_m = now
            print(f"[Mode] -> {sel_mode.upper()}")

        # B — toggle blink
        elif key == ord('b') and now - last_b > 0.5:
            blinker.enabled = not blinker.enabled
            keyboard_gui.status = (f"Blink selection: "
                                   f"{'ON' if blinker.enabled else 'OFF'}")
            last_b = now
            print(f"[Blink] {'ON' if blinker.enabled else 'OFF'}")

        # D -- toggle EAR debug logging
        elif key == ord('d'):
            blinker.debug_ear = not blinker.debug_ear
            state = 'ON' if blinker.debug_ear else 'OFF'
            keyboard_gui.status = f"EAR debug: {state}"
            print(f"[Debug] EAR logging {state}")

        # C — calibrate / load saved
        elif key == ord('c') and hc is not None:

            if not (l_locked and r_locked):
                # ── First C press: lock spheres ────────────────────────────
                saved = cal_mgr.load()
                if saved:
                    l_off = saved['left_offset']
                    r_off = saved['right_offset']
                    l_sc  = saved['left_scale']
                    r_sc  = saved['right_scale']
                    l_locked = r_locked = True
                    if saved['homography'] is not None:
                        calib.load_homography(saved['homography'])
                    print("[Calib] Loaded saved calibration.")
                    keyboard_gui.status = ("Saved calibration loaded. "
                                           "Look straight ahead, press C again.")
                else:
                    cns   = compute_scale(npts)
                    cdl   = Rf.T @ np.array([0., 0., 1.])
                    l_off = Rf.T @ (i3l - hc) + BASE_RADIUS * cdl
                    r_off = Rf.T @ (i3r - hc) + BASE_RADIUS * cdl
                    l_sc  = r_sc = cns
                    l_locked = r_locked = True
                    print("[Calib] Eye spheres locked (fresh).")
                    keyboard_gui.status = ("Spheres locked. "
                                           "1→TL  2→TR  3→BL  4→BR + SPACE.")

            # ── Always set monitor plane on C press ────────────────────────
            if l_locked and sw_l is None:
                # compute initial sphere positions
                swl_tmp = hc + Rf @ l_off
                swr_tmp = hc + Rf @ r_off
            else:
                swl_tmp = sw_l if sw_l is not None else hc + Rf @ l_off
                swr_tmp = sw_r if sw_r is not None else hc + Rf @ r_off

            fwd_h = normalize(normalize(i3l - swl_tmp)
                              + normalize(i3r - swr_tmp))
            go_h  = (swl_tmp + swr_tmp) * 0.5

            mon_corners, mon_cw, mon_nw, upc = create_monitor_plane(
                hc, Rf, lms, fw, fh,
                fwd=fwd_h, go=go_h, gd=fwd_h,
            )
            orbit['frozen'] = True
            orbit['pivot']  = mon_cw.copy()
            gf.reset(); fix_det.reset()
            print("[Calib] Monitor plane set.")
            keyboard_gui.status = ("Monitor plane set. "
                                   "1→TL  2→TR  3→BL  4→BR + SPACE each.")

        # 1 / 2 / 3 / 4 — start a corner calibration step
        elif (key in (ord('1'), ord('2'), ord('3'), ord('4'))
              and l_locked and mon_corners is not None):
            idx = key - ord('1')
            calib.step   = idx
            calib.active = True
            calib._all.clear()
            calib._stable = []
            print(f"[Calib] Look at {calib.LABELS[idx]}. Press SPACE.")
            keyboard_gui.status = f"Look at {calib.LABELS[idx]}, press SPACE."

        # SPACE — confirm calibration point
        elif key == ord(' ') or (KB_OK
                                  and kb.is_pressed('space')
                                  and not prev_space):
            if calib.active:
                ok = calib.confirm_point()
                if ok and not calib.active:
                    # Save everything
                    cal_mgr.save(l_off, r_off, l_sc, r_sc, calib._H)
                    keyboard_gui.status = "Calibration saved!  Type with your eyes."
                elif ok:
                    keyboard_gui.status = (f"Look at {calib.current_label}, "
                                           f"press SPACE.")
                else:
                    keyboard_gui.status = ("Need more stable frames — "
                                           "hold your gaze still.")

        # X — drop gaze debug marker
        elif (key == ord('x')
              and mon_corners is not None
              and avg_dir is not None
              and sw_l is not None):
            O3 = (sw_l + sw_r) * 0.5
            ab = ray_plane_ab(O3, avg_dir, mon_corners, mon_cw, mon_nw)
            if ab:
                a_, b_ = ab
                a_ = 1.0 - a_;  b_ = 1.0 - b_
                gaze_markers.append((a_, b_))
                print(f"[Marker] a={a_:.3f}  b={b_:.3f}")

        # Space debounce
        if not (KB_OK and kb.is_pressed('space')):
            prev_space = False
        elif key == ord(' '):
            prev_space = True

    # ── Shutdown ───────────────────────────────────────────────────────────
    collector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[EyeWave] Session ended.  "
          f"{collector.count} gaze samples saved → {GAZE_DATA_FILE}")


if __name__ == "__main__":
    main()
