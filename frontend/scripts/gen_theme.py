"""Generate light + dark theme CSS variables (as OKLCH `L C H` triples) and the
Tailwind colors object from the existing palette. Run once; paste output into
index.css and tailwind.config.js. Kept in-repo so the palette can be regenerated.
"""
import math
import re

# --- color math ----------------------------------------------------------------

def _srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def hex_to_oklch(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    r, g, b = _srgb_to_lin(r), _srgb_to_lin(g), _srgb_to_lin(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    C = math.hypot(a, bb)
    H = math.degrees(math.atan2(bb, a)) % 360
    return (round(L, 4), round(C, 4), round(H, 2))

def parse_oklch(s):
    nums = re.findall(r"[-\d.]+", s)
    L, C, H = float(nums[0]), float(nums[1]), float(nums[2]) if len(nums) > 2 else 0.0
    return (round(L, 4), round(C, 4), round(H, 2))

def to_lch(v):
    return hex_to_oklch(v) if v.startswith("#") else parse_oklch(v)

# --- light palette (verbatim from tailwind.config.js) --------------------------

LIGHT = {
    "surface": "oklch(1 0 0)", "warning-bg": "oklch(0.96 0.04 90)",
    "on-surface-variant": "#444655", "on-error-container": "#93000a",
    "on-tertiary-container": "#fffbff", "on-error": "#ffffff", "primary": "#264dd9",
    "on-secondary-container": "#fffbff", "success-fg": "oklch(0.38 0.08 150)",
    "tertiary-fixed-dim": "#ffb68e", "outline": "#747686", "surface-tint": "#294fdb",
    "ink-4": "oklch(0.68 0.009 250)", "primary-fixed-dim": "#b8c3ff",
    "primary-container": "#4568f3", "tertiary-container": "#be5600",
    "surface-container-high": "#e8e8ee", "hairline": "oklch(0.94 0.005 250)",
    "secondary-fixed": "#e1e0ff", "on-secondary": "#ffffff",
    "on-secondary-fixed-variant": "#2f2ebe", "error-container": "#ffdad6",
    "secondary-fixed-dim": "#c0c1ff", "surface-dim": "#d9d9df",
    "bg-tint": "oklch(0.975 0.004 250)", "accent-bg": "oklch(0.96 0.025 265)",
    "accent": "oklch(0.5 0.13 265)", "ink-2": "oklch(0.36 0.012 250)",
    "secondary": "#4648d4", "border-2": "oklch(0.86 0.006 250)",
    "surface-container-low": "#f3f3f9", "tertiary-fixed": "#ffdbc9",
    "inverse-primary": "#b8c3ff", "error": "#ba1a1a", "on-secondary-fixed": "#07006c",
    "bg": "oklch(0.985 0.003 250)", "on-tertiary": "#ffffff",
    "on-tertiary-fixed-variant": "#763300", "background": "#f9f9ff",
    "border": "oklch(0.92 0.005 250)", "outline-variant": "#c4c5d7",
    "warning-fg": "oklch(0.4 0.07 75)", "secondary-container": "#6063ee",
    "surface-variant": "#e2e2e8", "ink-3": "oklch(0.52 0.011 250)",
    "on-primary": "#ffffff", "on-primary-fixed": "#001355",
    "on-primary-fixed-variant": "#0035bd", "surface-container-lowest": "#ffffff",
    "on-background": "#1a1c20", "tertiary": "#974300",
    "on-primary-container": "#fffbff", "success-bg": "oklch(0.95 0.045 150)",
    "inverse-surface": "#2f3035", "on-surface": "#1a1c20",
    "surface-2": "oklch(0.965 0.004 250)", "ink": "oklch(0.22 0.012 250)",
    "primary-fixed": "#dde1ff", "on-tertiary-fixed": "#331200",
    "surface-container": "#ededf3", "surface-container-highest": "#e2e2e8",
    "inverse-on-surface": "#f0f0f6", "surface-bright": "#f9f9ff",
}

# --- dark palette: explicit values for tokens that actually show in the UI ------
# Authored as OKLCH triples (L C H). Neutrals share hue 265 for a cool slate feel.
DARK_OVERRIDES = {
    # app + panel surfaces (dark, low chroma, ascending elevation)
    "bg": (0.18, 0.006, 265), "background": (0.18, 0.006, 265),
    "bg-tint": (0.205, 0.006, 265), "surface": (0.235, 0.007, 265),
    "surface-2": (0.235, 0.007, 265), "surface-bright": (0.27, 0.008, 265),
    "surface-container-lowest": (0.2, 0.006, 265),
    "surface-container-low": (0.225, 0.007, 265),
    "surface-container": (0.275, 0.008, 265),
    "surface-container-high": (0.31, 0.008, 265),
    "surface-container-highest": (0.34, 0.008, 265),
    "surface-variant": (0.32, 0.008, 265), "surface-dim": (0.17, 0.006, 265),
    # text
    "ink": (0.95, 0.004, 265), "ink-2": (0.86, 0.006, 265),
    "ink-3": (0.72, 0.008, 265), "ink-4": (0.6, 0.008, 265),
    "on-background": (0.95, 0.004, 265), "on-surface": (0.95, 0.004, 265),
    # borders / hairlines (light-on-dark, subtle)
    "hairline": (0.31, 0.006, 265), "border": (0.34, 0.006, 265),
    "border-2": (0.37, 0.006, 265), "outline": (0.6, 0.008, 265),
    "outline-variant": (0.42, 0.007, 265),
    # brand / accents (brightened for contrast on dark)
    "primary": (0.64, 0.17, 264), "surface-tint": (0.64, 0.17, 264),
    "primary-container": (0.58, 0.16, 264), "primary-fixed-dim": (0.78, 0.1, 264),
    "accent": (0.72, 0.12, 265), "accent-bg": (0.34, 0.05, 265),
    "secondary": (0.62, 0.14, 280), "secondary-container": (0.58, 0.14, 280),
    # semantic
    "error": (0.68, 0.19, 25), "error-container": (0.32, 0.09, 25),
    "on-error-container": (0.86, 0.1, 22), "on-error": (0.99, 0, 0),
    "success-fg": (0.74, 0.14, 155), "success-bg": (0.3, 0.05, 155),
    "warning-fg": (0.8, 0.12, 85), "warning-bg": (0.32, 0.05, 85),
    # text that sits on brand fills stays near-white
    "on-primary": (0.99, 0, 0), "on-secondary": (0.99, 0, 0),
    "on-tertiary": (0.99, 0, 0),
}

def dark_for(name, light_lch):
    if name in DARK_OVERRIDES:
        return DARK_OVERRIDES[name]
    L, C, H = light_lch
    if C < 0.03:  # neutral -> invert lightness around mid
        return (round(min(0.97, max(0.16, 1.04 - L)), 4), C, H)
    return (round(min(0.85, L + 0.15), 4), C, H)  # chromatic -> lighten a touch

def fmt(lch):
    L, C, H = lch
    return f"{L} {C} {H}"

light_lch = {k: to_lch(v) for k, v in LIGHT.items()}
dark_lch = {k: dark_for(k, light_lch[k]) for k in LIGHT}

print("/* ---- :root (light) ---- */")
for k in LIGHT:
    print(f"  --c-{k}: {fmt(light_lch[k])};")
print("\n/* ---- .dark ---- */")
for k in LIGHT:
    print(f"  --c-{k}: {fmt(dark_lch[k])};")
print("\n// ---- tailwind colors ---- ")
for k in LIGHT:
    print(f'        "{k}": "oklch(var(--c-{k}) / <alpha-value>)",')
