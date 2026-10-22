"""
historical_degradation.py
=========================
Train-time photometric augmentation that degrades crisp modern panchromatic
tiles so they resemble 1960s-70s scanned aerial film. Forces the model toward
invariance to the low-level statistics that differ between modern and historical
imagery. PHOTOMETRIC ONLY (never moves a pixel), so labels stay valid. Operates
in raw [0,255] float32 end to end -- never divides by 255.
"""
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

try:
    import cv2
    def _blur(x, s):
        return x if s <= 0 else cv2.GaussianBlur(x, (0, 0), s, s, borderType=cv2.BORDER_REFLECT)
    def _resize(x, hw, up):
        H, W = hw
        return cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR if up else cv2.INTER_AREA)
except ImportError:
    from scipy.ndimage import gaussian_filter, zoom
    def _blur(x, s):
        return x if s <= 0 else gaussian_filter(x, s, mode="reflect")
    def _resize(x, hw, up):
        return zoom(x, (hw[0]/x.shape[0], hw[1]/x.shape[1]), order=1)


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


@dataclass
class DegradeConfig:
    p_apply: float = 0.8
    p_blur: float = 0.7;      sigma_lo: float = 0.5;  sigma_hi: float = 2.0
    p_resample: float = 0.5;  scale_lo: float = 0.45; scale_hi: float = 0.80
    p_vignette: float = 0.4;  vig_lo: float = 0.05;   vig_hi: float = 0.30
    p_gamma: float = 0.7;     gamma_lo: float = 0.7;  gamma_hi: float = 1.5
    p_tone: float = 0.7;      contrast_lo: float = 0.60; contrast_hi: float = 1.0
    black_lift_max: float = 35.0; white_pull_max: float = 35.0; bright_abs: float = 15.0
    p_grain: float = 0.8;     grain_lo: float = 3.0;  grain_hi: float = 12.0
    grain_sigma_lo: float = 0.5; grain_sigma_hi: float = 1.0; lum_couple: float = 0.5
    p_scan: float = 0.5;      scan_lo: float = 1.0;   scan_hi: float = 4.0


class HistoricalDegradation:
    def __init__(self, config: Optional[DegradeConfig] = None,
                 references: Optional[Sequence[np.ndarray]] = None, seed=None):
        self.c = config or DegradeConfig()
        self.refs = list(references) if references else []
        self.rng = np.random.default_rng(seed)
        try:
            from skimage.exposure import match_histograms
            self._match = match_histograms
        except ImportError:
            self._match = None

    def __call__(self, img, is_modern=True):
        if not is_modern:
            return img
        had_ch = img.ndim == 3
        x = (img[..., 0] if had_ch else img).astype(np.float32)
        c, rng = self.c, self.rng
        if rng.random() < c.p_apply:
            if rng.random() < c.p_blur:
                x = _blur(x, _u(rng, c.sigma_lo, c.sigma_hi))
            if rng.random() < c.p_resample:
                H, W = x.shape; s = _u(rng, c.scale_lo, c.scale_hi)
                x = _resize(_resize(x, (max(1,int(H*s)), max(1,int(W*s))), False), (H, W), True)
            if rng.random() < c.p_vignette:
                H, W = x.shape; strength = _u(rng, c.vig_lo, c.vig_hi)
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                cy, cx = (H-1)/2, (W-1)/2
                r = np.sqrt(((yy-cy)/cy)**2 + ((xx-cx)/cx)**2); r /= r.max()+1e-8
                x = x * (1.0 - strength*(r**2))
            if rng.random() < c.p_gamma:
                g = _u(rng, c.gamma_lo, c.gamma_hi)
                x = np.power(np.clip(x/255.0, 0, 1), g) * 255.0
            if rng.random() < c.p_tone:
                con = _u(rng, c.contrast_lo, c.contrast_hi)
                lo = _u(rng, 0, c.black_lift_max); hi = 255.0 - _u(rng, 0, c.white_pull_max)
                b = _u(rng, -c.bright_abs, c.bright_abs)
                x = (x - 127.5)*con + 127.5 + b
                x = lo + (x/255.0)*(hi-lo)
            if self.refs and self._match is not None and rng.random() < 0.4:
                x = self._match(x, self.refs[rng.integers(len(self.refs))].astype(np.float32)).astype(np.float32)
            if rng.random() < c.p_grain:
                H, W = x.shape
                std = _u(rng, c.grain_lo, c.grain_hi); gs = _u(rng, c.grain_sigma_lo, c.grain_sigma_hi)
                noise = _blur(rng.standard_normal((H, W)).astype(np.float32), gs)
                noise /= noise.std()+1e-8
                x01 = np.clip(x/255.0, 0, 1); env = 4.0*x01*(1-x01)
                amp = (1-c.lum_couple) + c.lum_couple*env
                x = x + noise*std*amp
            if rng.random() < c.p_scan:
                x = x + rng.standard_normal(x.shape).astype(np.float32)*_u(rng, c.scan_lo, c.scan_hi)
        x = np.clip(x, 0, 255).astype(np.float32)
        return x[..., None] if had_ch else x
