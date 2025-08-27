# Signal Detector for RavenPro

This program automatically generates a **RavenPro selection table** identifying possible calls from audio recordings based on amplitude. It supports multiple audio channels and can be configured to detect calls within specific frequency ranges.

> ⚠️ **Note:** The program may produce errors in noisy data. Manual verification of detected calls is recommended.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Input / Output](#input--output)
- [Known Issues & Limitations](#known-issues--limitations)
- [Changelog](#changelog)

## Overview

- Detects signals in audio files
- Generates a **RavenPro-compatible selection table** (`.txt`)
- Filters calls to a default frequency range of **13–27 Hz**
- Improved detection for low-frequency and clustered calls

## Installation

1. Ensure **Python 3.x** is installed
2. Install required dependencies:
   ```bash
   pip install numpy scipy
   ```
3. Prepare your audio files in `.wav` or `.aif` format

## Configuration

All settings are configured in `whale_detector_config.txt`.

### Channels
- `bad_channels`: List channels to exclude from detection

### Detection Parameters
- `threshold`: Sensitivity of call detection by amplitude; lower to decrease sensitivity (numeric, no units)
- `freq_low`: Lower bound of frequency detection (Hz)
- `freq_high`: Upper bound of frequency detection (Hz)

### Example Configuration
```
# Channel settings (1-based indexing, separate multiple channels with commas, in order)
bad_channels = 1. 4

# Detection parameters 
# To detect fainter calls, decrease threshold and vice versa
threshold = 3.0
freq_low = 12.0
freq_high = 32.0

# Call duration range (seconds)
min_duration = 0.5
max_duration = 3

# Advanced parameters (optional)
merge_window = 1.5
min_bandwidth = 3.0 
```

## Usage

1. Edit `signal_detector_config.txt` with your desired settings
2. Run the detector:
   ```bash
   python signal_detector.py
   ```
3. Check the output directory for generated selection tables

## Input / Output

- **Input:** Directory containing audio files (`.wav` or `.aif`)
- **Output:** Directory where selection table text files will be saved

> **Note:** Do **not** include units in numeric inputs. Example: Use `6.0`, **not** `6.0s`.

## Known Issues & Limitations

- May miss low-amplitude or clustered calls
- Calls mixed with other overlapping calls may not be detected
- High noise levels may produce false positives or missed detections

## Changelog

### June 11, 2025
- Removed SNR filtering
- Calls automatically detected in **13–27 Hz range**
- Improved low-frequency detection
