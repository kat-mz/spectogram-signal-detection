import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, hilbert
import os
from scipy.ndimage import uniform_filter1d #Commented out for now, June 10th, 2025

class SignalDetector:
    def __init__(self, 
                 freq_range,
                 call_duration_range,
                 threshold_factor,
                 merge_window_sec):
      
        self.freq_range = freq_range
        self.call_duration_range = call_duration_range
        self.threshold_factor = threshold_factor
        self.merge_window_sec = merge_window_sec

    def bandpass_filter(self, data, sr, order=4):
        lowcut, highcut = self.freq_range
        sos = butter(order, [lowcut, highcut], btype='band', fs=sr, output='sos')
        return sosfilt(sos, data)


    def detect_calls(self, signal, sr):
        min_dur, max_dur = self.call_duration_range
        
        # Get envelope using Hilbert transform analytical signal
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        
        # Percentile-based threshold, 
        base_threshold = np.percentile(envelope, 80)  
        
        #Creates bool array of whether instantaneous amplitude is above threshold at that time
        above_base = envelope > base_threshold

        #Converts to array of 0,1
        diff = np.diff(above_base.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases, end and start of the audiofile
        if above_base[0]:
            starts = np.concatenate([[0], starts])
        if above_base[-1]:
            ends = np.concatenate([ends, [len(above_base)]])
        
        events = []
        
        for start, end in zip(starts, ends):
            dur = (end - start) / sr
            
            # Check duration
            if not (min_dur <= dur <= max_dur):
                continue
            
            # Find peak
            peak_idx = np.argmax(envelope[start:end]) + start
            call_amplitude = envelope[peak_idx]
            
            # Simplified noise estimation - use local background
            window_size = int(5 * sr)  #5 second window
            noise_start = max(0, peak_idx - window_size)
            noise_end = min(len(envelope), peak_idx + window_size)
            
            # Exclude the call region itself
            call_buffer = int(0.05 * sr)  # 0.05 second buffer around call
            noise_region = np.concatenate([
                envelope[noise_start:max(noise_start, peak_idx - call_buffer)],
                envelope[min(len(envelope), peak_idx + call_buffer):noise_end]
            ])
            
            if len(noise_region) == 0:
                continue
                
            noise_level = np.percentile(noise_region, 50)  # Use median instead of mean
            
            # Calculate noise threshold (e.g., 2.5x the background noise level)
            noise_threshold = noise_level * 2.5  # You can adjust this multiplier
            
            # Find how far we can extend before the call before hitting noise threshold
            before_noise_time = 0.0
            search_start = max(0, start - int(3 * sr))  # Search up to 3 seconds before
            for i in range(start - 1, search_start - 1, -1):
                if i < 0:
                    before_noise_time = (start - search_start) / sr
                    break
                if envelope[i] > noise_threshold:
                    before_noise_time = (start - i) / sr
                    break
            else:
                # Reached the search limit without hitting threshold
                before_noise_time = min(2.0, (start - search_start) / sr)
        
            # Find how far we can extend after the call before hitting noise threshold
            after_noise_time = 0.0
            search_end = min(len(envelope), end + int(3 * sr))  # Search up to 2 seconds after
            for i in range(end, search_end):
                if i >= len(envelope):
                    after_noise_time = (search_end - end) / sr
                    break
                if envelope[i] > noise_threshold:
                    after_noise_time = (i - end) / sr
                    break
            else:
                # Reached the search limit without hitting threshold
                 after_noise_time = (search_end - end) / sr
            
            # FFT of individual call segment (windowing optional)
            segment = signal[start:end]
            windowed = segment * np.hanning(len(segment))
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

            # Restrict to desired frequency band
            in_band = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
            freqs_band = freqs[in_band]
            spectrum_band = spectrum[in_band]
            
            threshold = np.percentile(spectrum_band, 5)
            significant_freqs = freqs_band[spectrum_band > threshold]

            max_freq = np.max(significant_freqs)
            min_freq = np.min(significant_freqs)

            # Calculate SNR
            snr = call_amplitude / (noise_level + 1e-10)
            
            # More lenient threshold check
            if snr >= self.threshold_factor:
                event = {
                    "start": start / sr,
                    "end": end / sr,
                    "peak": peak_idx / sr,
                    "duration": dur,
                    "snr": snr,
                    "call_amplitude": call_amplitude,
                    "noise_level": noise_level,
                    "noise_threshold": noise_threshold,
                    "before_noise": before_noise_time,  # Now in seconds
                    "after_noise": after_noise_time,   # Now in seconds
                    "max_freq": max_freq,
                    "min_freq": min_freq
                }
                events.append(event)

        if not events:
            return []

        # Merge nearby detections
        events.sort(key=lambda e: e["peak"])
        merged = [events[0]]
        
        for event in events[1:]:
            if event["peak"] - merged[-1]["peak"] < self.merge_window_sec:
                if event["snr"] > merged[-1]["snr"]:
                    merged[-1] = event
            else:
                merged.append(event)

        return merged

    def process_audio_file(self, filename, bad_channels=None):
        if bad_channels is None:
            bad_channels = []
            
        try:
            data, sr = sf.read(filename)
        except Exception as e:
            raise ValueError(f"Could not read audio file {filename}: {e}")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        print(f"Loaded audio: {data.shape[1]} channels, {data.shape[0]} samples, {sr} Hz")
        print(f"Duration: {data.shape[0] / sr:.2f} seconds")

        detections = []
        for ch in range(data.shape[1]):
            if ch in bad_channels:
                print(f"Skipping bad channel {ch + 1}")
                continue
                
            print(f"Processing channel {ch + 1}...")
            signal = data[:, ch]
            filtered = self.bandpass_filter(signal, sr)
            events = self.detect_calls(filtered, sr)
            
            for event in events:
                event["channel"] = ch
                detections.append(event)
            
            print(f"  Found {len(events)} potential calls")
            if events:
                snrs = [e['snr'] for e in events]
                print(f"  SNR range: {np.min(snrs):.2f} - {np.max(snrs):.2f}")

        return detections

    def save_detections(self, detections, output_path, input_filename):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", newline='', encoding='utf-8') as f:
            headers = [
                "Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                "Low Freq (Hz)", "High Freq (Hz)", "Begin File", "File Offset (s)",
                "Call Type", "Call ID",  "Before Noise (s)", "After Noise (s)", "Notes"
                ]
            f.write('\t'.join(headers) + '\n')

            os.path.basename(input_filename)

            for i, d in enumerate(detections, 1):
                row_data = [
                    str(i), "1", str(d["channel"] + 1),
                    f"{d['start']:.9f}", f"{d['end']:.9f}",
                    f"{d['min_freq']:.9f}", f"{d['max_freq']:.9f}", "", 
                    "", "f20p", "", "", "", ""

                    #f"{d['before_noise']:.9f}
                    #f"{d['after_noise']:.9f
                ]
                f.write('\t'.join(row_data) + '\n')

    def print_summary(self, detections):
        if not detections:
            print("No fin whale calls detected.")
            return

        durations = np.array([d['duration'] for d in detections])

        print(f"\nDetection Summary:")
        print(f"- Total calls detected: {len(detections)}")
        print(f"- Call duration (s): Min:{np.min(durations):.2f}  Mean:{np.mean(durations):.2f}  Max:{np.max(durations):.2f}")


def load_parameters(txt_file):
    params = {}
    
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    try:
        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key = value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle different data types
                    if key in ['threshold', 'freq_low', 'freq_high', 'min_duration', 'max_duration', 'merge_window']:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            print(f"Warning: Invalid float value for {key} on line {line_num}: {value}")
                    elif key == 'bad_channels':
                        if value:
                            try:
                                # Handle comma-separated values
                                params[key] = [int(ch.strip()) - 1 for ch in value.split(',') if ch.strip()]  # Convert to 0-based
                            except ValueError:
                                print(f"Warning: Invalid channel numbers on line {line_num}: {value}")
                                params[key] = []
                        else:
                            params[key] = []
                    else:
                        params[key] = value
    
    except Exception as e:
        raise ValueError(f"Error reading text file: {e}")
    
    return params


def main():
    # Hardcoded path to text file in the same folder as the program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file = os.path.join(script_dir, "whale_detector_param.txt")
    
    print(f"Looking for text file at: {txt_file}")
    
    # Load parameters from text file
    params = load_parameters(txt_file)
    
    # Check required parameters
    if 'input_file' not in params:
        raise ValueError("Text file must specify 'input_file'")
    
    input_file = params['input_file']
    output_folder = params.get('output_folder', '.')
    output_filename = params.get('output_filename', 'fin_whale_detections.txt')
    output_path = os.path.join(output_folder, output_filename)
    
    # Get parameters from text file - use values directly
    freq_low = params['freq_low']
    freq_high = params['freq_high'] 
    min_duration = params['min_duration']
    max_duration = params['max_duration']
    threshold = params['threshold']
    merge_window = params['merge_window']
    bad_channels = params.get('bad_channels', [])
    
    print(f"Using frequency range: {freq_low} - {freq_high} Hz")
    print(f"Using threshold: {threshold}")
    print(f"Using duration range: {min_duration} - {max_duration} s")
    
    # Initialize detector with parameters from text file
    detector = FinWhaleDetector(
        freq_range=(freq_low, freq_high),
        call_duration_range=(min_duration, max_duration),
        threshold_factor=threshold,
        merge_window_sec=merge_window
    )
    
    try:
        print(f"Processing {input_file}...")
        if bad_channels:
            print(f"Ignoring channels: {[ch + 1 for ch in bad_channels]}")
        
        detections = detector.process_audio_file(input_file, bad_channels)
        detector.save_detections(detections, output_path, input_file)
        detector.print_summary(detections)
        
        print(f"\nResults saved to {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
