"""
QUANTUM EYE — BEYOND RANDOMNESS
Configuration and Theme Palette
"""

# Visual Theme Colors
THEME = {
    "bg_main": "#f8fafc",        # Very light slate/white
    "bg_card": "#ffffff",        # Pure white card
    "bg_sidebar": "#0f172a",     # Deep navy sidebar for clear scientific contrast
    "sidebar_text": "#e2e8f0",
    "text_primary": "#0f172a",   # Navy blue typography
    "text_secondary": "#475569", # Slate text
    "text_muted": "#64748b",
    "border": "#e2e8f0",         # Light subtle border
    "primary": "#0284c7",        # Electric blue / cyan
    "accent_cyan": "#06b6d4",    # Cyan highlight
    "accent_teal": "#0d9488",    # Teal highlight
    "accent_violet": "#8b5cf6",  # Violet accent
    "accent_blue": "#2563eb",    # Royal blue
    "success": "#10b981",        # Emerald
    "warning": "#f59e0b",        # Amber
    "danger": "#ef4444",         # Red
    "plot_palette": ["#0284c7", "#0d9488", "#8b5cf6", "#f59e0b", "#ec4899", "#10b981", "#6366f1"]
}

# Application Metadata
APP_NAME = "QUANTUM EYE"
APP_SUBTITLE = "BEYOND RANDOMNESS"
APP_TAGLINE = "Machine Learning for Intelligent Complex Signal Analysis"
APP_VERSION = "2.0.0-PROTOTYPE"

# Benchmarking & Limits
MAX_UPLOAD_SIZE_MB = 100
DEFAULT_SAMPLING_RATE = 100.0  # Hz
MAX_POINTS_INTERACTIVE_PLOT = 15000  # Downsample for ultra-smooth Plotly rendering if signal is huge

