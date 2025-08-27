## Signal Detection Formulated for RavenPro

# How to use (required manual inputs)
- No inputs have units (e.g. don't enter 6.0s, just 6.0)
-----------------------------------------------------------------
[Input/Output Settings]
- Input full paths of directory for input file (.wav/aif) 
- Input full path of directory for location of directory for outputted text file

Ending slashes are NOT needed
-----------------------------------------------------------------
# Channel Settings
- bad_channels: input the channels that you do not want to include
- detection parameters:
    threshold: input the variability of a call
    freq_low: lower bound of bandwidth
    freq_high: higher bound of bandwidth
-----------------------------------------------------------------

# Current Errors/Updates Needed as of June 11th, 2025
- Associating calls, such that CALL IDS are in order
- Removing SNR (FIXED)
- Very error prone in high noise areas *
- Misses low amplitude sounds, clustered sounds *
- For calls mixed in with other calls - doesn't detect due to length parameters

#Update (1) - June 11th, 2025
- Removed SNR
- Automated range of calls to be around 13 HZ to 27 HZ - sounds are now clipped at those heights
- Better low frequency detection
