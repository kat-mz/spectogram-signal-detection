import numpy as np
import os
from collections import defaultdict

class CallIDAssigner:
    def __init__(self, time_tolerance=0.5, min_channels=2):
        """
        Initialize the Call ID assigner.
        
        Args:
            time_tolerance (float): Maximum time difference (seconds) to consider calls as the same
            min_channels (int): Minimum number of channels required for a valid call group
        """
        self.time_tolerance = time_tolerance
        self.min_channels = min_channels
    
    def load_detections(self, input_file):
        """Load detections from the output file of the whale detector."""
        detections = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Skip header
                header = lines[0].strip().split('\t')
                
                for line_num, line in enumerate(lines[1:], 2):
                    if not line.strip():
                        continue
                        
                    parts = line.strip().split('\t')
                    if len(parts) < len(header):
                        print(f"Warning: Skipping malformed line {line_num}")
                        continue
                    
                    try:
                        detection = {
                            'selection': int(parts[0]),
                            'view': parts[1],
                            'channel': int(parts[2]),
                            'begin_time': float(parts[3]),
                            'end_time': float(parts[4]),
                            'low_freq': float(parts[5]),
                            'high_freq': float(parts[6]),
                            'call_id': parts[7],  # Currently empty
                            'call_type': parts[8],
                            'before_noise': parts[9],
                            'after_noise': parts[10],
                            'notes': parts[11] if len(parts) > 11 else '',
                            'peak_time': (float(parts[3]) + float(parts[4])) / 2  # Calculate peak time
                        }
                        detections.append(detection)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error parsing line {line_num}: {e}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except Exception as e:
            raise ValueError(f"Error reading input file: {e}")
        
        return detections, header
    
    def group_detections_by_time(self, detections):
        """Group detections that occur at similar times across channels."""
        # Sort detections by peak time
        sorted_detections = sorted(detections, key=lambda x: x['peak_time'])
        
        groups = []
        used_indices = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue
            
            # Start a new group with this detection
            group = [detection]
            used_indices.add(i)
            group_channels = {detection['channel']}
            
            # Find other detections within time tolerance
            for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                time_diff = abs(other_detection['peak_time'] - detection['peak_time'])
                
                # If time difference is too large, stop searching (since list is sorted)
                if time_diff > self.time_tolerance:
                    break
                
                # Check if this detection is from a different channel
                if other_detection['channel'] not in group_channels:
                    group.append(other_detection)
                    used_indices.add(j)
                    group_channels.add(other_detection['channel'])
        
            groups.append(group)
        
        return groups
    
    def assign_call_ids(self, detections):
        """Assign Call IDs to detections based on cross-channel correlation."""
        print(f"Processing {len(detections)} detections...")
        
        # Group detections by time
        groups = self.group_detections_by_time(detections)
        
        print(f"Found {len(groups)} potential call groups")
        
        # Filter groups that have minimum required channels
        valid_groups = []
        for group in groups:
            if len(group) >= self.min_channels:
                valid_groups.append(group)
        
        print(f"Found {len(valid_groups)} valid call groups (>= {self.min_channels} channels)")
        
        # Assign Call IDs to valid groups
        call_id_map = {}  # Maps (selection_num) to call_id
        
        for call_id, group in enumerate(valid_groups, 1):
            print(f"Call ID {call_id}: {len(group)} detections at ~{group[0]['peak_time']:.2f}s")
            print(f"  Channels: {sorted([d['channel'] for d in group])}")
            
            for detection in group:
                call_id_map[detection['selection']] = str(call_id)
        
        # Apply Call IDs to original detections
        for detection in detections:
            selection_num = detection['selection']
            if selection_num in call_id_map:
                detection['call_id'] = call_id_map[selection_num]
            else:
                detection['call_id'] = ''  # Keep empty for detections not in valid groups
        
        return detections, len(valid_groups)
    
    def save_updated_detections(self, detections, header, output_file):
        """Save updated detections with Call IDs to file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write('\t'.join(header) + '\n')
            
            # Write detections
            for detection in detections:
                row_data = [
                    str(detection['selection']),
                    detection['view'],
                    str(detection['channel']),
                    f"{detection['begin_time']:.9f}",
                    f"{detection['end_time']:.9f}",
                    f"{detection['low_freq']:.1f}",
                    f"{detection['high_freq']:.1f}",
                    detection['call_id'],
                    detection['call_type'],
                    detection['before_noise'],
                    detection['after_noise'],
                    detection['notes']
                ]
                f.write('\t'.join(row_data) + '\n')
    
    def print_summary(self, detections, num_call_groups):
        """Print summary of Call ID assignment."""
        total_detections = len(detections)
        assigned_detections = len([d for d in detections if d['call_id']])
        unassigned_detections = total_detections - assigned_detections
        
        print(f"\nCall ID Assignment Summary:")
        print(f"- Total detections: {total_detections}")
        print(f"- Detections with Call IDs: {assigned_detections}")
        print(f"- Detections without Call IDs: {unassigned_detections}")
        print(f"- Number of call groups: {num_call_groups}")
        
        if num_call_groups > 0:
            # Show Call ID distribution
            call_id_counts = defaultdict(int)
            for detection in detections:
                if detection['call_id']:
                    call_id_counts[detection['call_id']] += 1
            
            print(f"\nCall ID Distribution:")
            for call_id in sorted(call_id_counts.keys(), key=int):
                count = call_id_counts[call_id]
                print(f"  Call ID {call_id}: {count} channels")


def load_parameters_for_assigner(txt_file):
    """Load parameters specific to the Call ID assigner."""
    params = {
        'time_tolerance': 0.5,  # Default values
        'min_channels': 2
    }
    
    if not os.path.exists(txt_file):
        print(f"Config file not found: {txt_file}. Using default parameters.")
        return params
    
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
                    
                    if key == 'time_tolerance':
                        try:
                            params['time_tolerance'] = float(value)
                        except ValueError:
                            print(f"Warning: Invalid time_tolerance value on line {line_num}: {value}")
                    elif key == 'min_channels':
                        try:
                            params['min_channels'] = int(value)
                        except ValueError:
                            print(f"Warning: Invalid min_channels value on line {line_num}: {value}")
    
    except Exception as e:
        print(f"Error reading config file: {e}. Using default parameters.")
    
    return params


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "call_id_assigner_param.txt")
    
    # Load parameters
    params = load_parameters_for_assigner(config_file)
    
    # Default input/output files (can be overridden in config)
    input_file = "fin_whale_detections.txt"  # Output from whale detector
    output_file = "fin_whale_detections_with_call_ids.txt"
    
    # Check if config file specifies input/output files
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'input_file':
                            input_file = value
                        elif key == 'output_file':
                            output_file = value
        except Exception as e:
            print(f"Warning: Error reading additional config parameters: {e}")
    
    print(f"Call ID Assigner Configuration:")
    print(f"- Input file: {input_file}")
    print(f"- Output file: {output_file}")
    print(f"- Time tolerance: {params['time_tolerance']} seconds")
    print(f"- Minimum channels required: {params['min_channels']}")
    
    # Initialize assigner
    assigner = CallIDAssigner(
        time_tolerance=params['time_tolerance'],
        min_channels=params['min_channels']
    )
    
    try:
        # Load detections
        print(f"\nLoading detections from {input_file}...")
        detections, header = assigner.load_detections(input_file)
        
        # Assign Call IDs
        print("\nAssigning Call IDs...")
        updated_detections, num_call_groups = assigner.assign_call_ids(detections)
        
        # Save updated file
        print(f"\nSaving updated detections to {output_file}...")
        assigner.save_updated_detections(updated_detections, header, output_file)
        
        # Print summary
        assigner.print_summary(updated_detections, num_call_groups)
        
        print(f"\nCall ID assignment complete! Updated file saved as: {output_file}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
