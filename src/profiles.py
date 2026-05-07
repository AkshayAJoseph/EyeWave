"""
profiles.py
============
User profile management for EyeWave.

Each profile stores tunable settings (EAR threshold, scan speeds, audio
preferences, etc.) as a named JSON file in the profiles/ directory.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.config import (
    BLINK_EAR_THRESH, BLINK_MIN_MS, BLINK_MAX_MS, BLINK_LONG_MAX_MS,
    SCAN_ROW_RATE, SCAN_COL_RATE, SCAN_COL_TIMEOUT,
    AUDIO_ENABLED, PROFILES_DIR,
)
    

@dataclass
class UserProfile:
    """All tunable settings that can be saved/loaded per user."""

    # Identity
    name: str = "default"

    # Blink detection
    ear_threshold: float = BLINK_EAR_THRESH
    blink_min_ms: int = BLINK_MIN_MS
    blink_max_ms: int = BLINK_MAX_MS
    blink_long_max_ms: int = BLINK_LONG_MAX_MS

    # Scanner
    scan_row_rate: float = SCAN_ROW_RATE
    scan_col_rate: float = SCAN_COL_RATE
    scan_col_timeout: float = SCAN_COL_TIMEOUT
    adaptive_speed: bool = True

    # Audio
    audio_enabled: bool = AUDIO_ENABLED

    # Calibration data (optional — stored from CalibrationManager)
    calibration: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        # Only pass known fields to avoid errors from stale profile files
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class ProfileManager:
    """Manages saving, loading, listing, and deleting user profiles."""

    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        self.current: UserProfile = UserProfile()
        self._profile_names: list[str] = []
        self._refresh_list()

    def _refresh_list(self):
        """Scan profiles directory for available profiles."""
        self._profile_names = []
        if os.path.isdir(self.profiles_dir):
            for f in sorted(os.listdir(self.profiles_dir)):
                if f.endswith('.json'):
                    self._profile_names.append(f[:-5])

    @property
    def available(self) -> list[str]:
        """List of available profile names."""
        self._refresh_list()
        return list(self._profile_names)

    def save(self, profile: UserProfile | None = None) -> str:
        """Save a profile to disk. Returns the file path."""
        p = profile or self.current
        path = os.path.join(self.profiles_dir, f"{p.name}.json")
        with open(path, 'w') as f:
            json.dump(p.to_dict(), f, indent=2)
        self._refresh_list()
        print(f"[Profile] Saved '{p.name}' -> {path}")
        return path

    def load(self, name: str) -> UserProfile | None:
        """Load a profile by name. Returns None if not found."""
        path = os.path.join(self.profiles_dir, f"{name}.json")
        if not os.path.exists(path):
            print(f"[Profile] '{name}' not found at {path}")
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.current = UserProfile.from_dict(data)
            print(f"[Profile] Loaded '{name}'")
            return self.current
        except Exception as e:
            print(f"[Profile] Error loading '{name}': {e}")
            return None

    def delete(self, name: str) -> bool:
        """Delete a profile by name."""
        path = os.path.join(self.profiles_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            self._refresh_list()
            print(f"[Profile] Deleted '{name}'")
            return True
        return False

    def load_default(self) -> UserProfile:
        """Load the 'default' profile, or create one if it doesn't exist."""
        p = self.load("default")
        if p is None:
            self.current = UserProfile(name="default")
            self.save()
        return self.current

    def cycle_next(self) -> UserProfile:
        """Cycle to the next available profile. Returns the loaded profile."""
        names = self.available
        if not names:
            return self.current
        try:
            idx = names.index(self.current.name)
            next_idx = (idx + 1) % len(names)
        except ValueError:
            next_idx = 0
        return self.load(names[next_idx]) or self.current

    def apply_to(self, blinker, scanner):
        """
        Apply the current profile's settings to live pipeline objects.

        Parameters
        ----------
        blinker : BlinkDetector
        scanner : ScanningController
        """
        p = self.current
        # Update the module-level config values used by blinker
        import src.config as cfg
        cfg.BLINK_EAR_THRESH  = p.ear_threshold
        cfg.BLINK_MIN_MS      = p.blink_min_ms
        cfg.BLINK_MAX_MS      = p.blink_max_ms
        cfg.BLINK_LONG_MAX_MS = p.blink_long_max_ms

        # Scanner settings
        cfg.SCAN_ROW_RATE     = p.scan_row_rate
        cfg.SCAN_COL_RATE     = p.scan_col_rate
        cfg.SCAN_COL_TIMEOUT  = p.scan_col_timeout
        scanner._row_rate     = p.scan_row_rate
        scanner._col_rate     = p.scan_col_rate
        scanner.adaptive      = p.adaptive_speed

        # Audio
        scanner.audio.enabled = p.audio_enabled

    def snapshot_from(self, blinker, scanner) -> UserProfile:
        """
        Capture current live settings into the current profile.

        Parameters
        ----------
        blinker : BlinkDetector
        scanner : ScanningController
        """
        import src.config as cfg
        p = self.current
        p.ear_threshold    = cfg.BLINK_EAR_THRESH
        p.blink_min_ms     = cfg.BLINK_MIN_MS
        p.blink_max_ms     = cfg.BLINK_MAX_MS
        p.blink_long_max_ms = cfg.BLINK_LONG_MAX_MS
        p.scan_row_rate    = scanner._row_rate
        p.scan_col_rate    = scanner._col_rate
        p.scan_col_timeout = cfg.SCAN_COL_TIMEOUT
        p.adaptive_speed   = scanner.adaptive
        p.audio_enabled    = scanner.audio.enabled
        return p
