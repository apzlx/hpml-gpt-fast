import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import re

class TransformerProfileAnalyzer:
    def __init__(self):
        # Define kernel patterns for each layer component
        self.kernel_patterns = {
            'attention.wqkv': [
                # INT8 patterns
                'triton_red_fused_bmm_4',
                'triton_per_fused__safe_softmax_add_index_logical_not_scalar_tensor_where_5',
                'triton_red_fused_bmm_6',
                'triton_red_fused_mm.*qkv',
                'triton_per_fused__softmax.*attention',
                # INT4 patterns
                'triton_poi_fused_bmm_2',
                'triton_red_fused_bmm_3',
                'triton_per_fused__softmax_add_index_logical_not_masked_fill_zeros_like_4',
                'void at::native::tinygemm_m16n8k16_chunk_kernel.*',
                'triton_red_fused_linear_qkv',
                'triton_poi_fused_attention.*',
                'triton_red_fused_matmul_qkv',
                'triton_red_fused_bmm_(4|6)',  # Make pattern more flexible
                'triton_per_fused__safe_softmax.*',
                'triton_poi_fused_stack',  # Add stack operations
                'triton_red_fused_mean_mul',  # Add mean operations
                'triton_red_fused_div_mm'  # Add division operations
            ],
            'attention.wo': [
                # INT8 patterns
                'triton_red_fused_bmm_7',
                'triton_red_fused__to_copy_add_bmm_mean_mul_rsqrt_13',
                'triton_red_fused_linear_wo',
                'triton_red_fused_matmul_wo',
                # INT4 patterns
                'triton_red_fused__to_copy_add_mean_mul_rsqrt_(10|11)',
                'triton_red_fused_bmm_5',
                'triton_red_fused__to_copy_add_mean_mul_rsqrt_10',
                'triton_poi_fused_linear_wo',
                'triton_red_fused_to_copy_add_attention_output',
                'triton_red_fused__to_copy_add_bmm_mm_mul_(13|15)',  # Make more flexible
                'triton_red_fused_mean_mul',
                'triton_poi_fused_index_put_stack'
            ],
            'feed_forward.w1': [
                # INT8 patterns
                'triton_red_fused_mm_9',
                'triton_red_fused_linear_gate',
                'triton_red_fused_matmul_w1',
                'triton_red_fused_gate_up',
                # INT4 patterns
                'triton_red_fused__to_copy_mean_mul_11',
                'triton_poi_fused_mul_silu_7',
                'triton_red_fused__to_copy_mean_mul',
                'triton_poi_fused_linear_gate',
                'triton_red_fused_gate_up_.*',
                'triton_red_fused__to_copy_add_bmm_mm_mul',
                'triton_red_fused_div_mm',
                'triton_red_fused_mm'
            ],
            'feed_forward.w2': [
                # INT8 patterns
                'triton_red_fused_add_bmm_mm_mul_14',
                'triton_red_fused_linear_proj',
                'triton_red_fused_matmul_w2',
                'triton_red_fused_proj_down',
                # INT4 patterns
                'triton_red_fused_add_bmm_mm_mul',
                'void at::native::tinygemm_m16n8k16_chunk_kernel(?!.*attention)',
                'triton_poi_fused_linear_proj',
                'triton_red_fused_proj_down_.*'
            ],
            'feed_forward.w3': [
                # INT8 patterns
                'triton_red_fused_mm_12',
                'triton_red_fused_linear_w3',
                'triton_red_fused_matmul_w3',
                # INT4 patterns
                'triton_red_fused__to_copy_add_mean_mul_rsqrt_13',
                'triton_poi_fused_index_put_1',
                'triton_poi_fused_linear_w3',
                'triton_red_fused_to_copy_add_w3'
            ]
        }
        
        # Add ordering information
        self.layer_sequence = [
            'attention.wqkv',
            'attention.wo',
            'feed_forward.w1',
            'feed_forward.w2',
            'feed_forward.w3'
        ]
        
        # Add debug logging for kernel matching
        self.debug = True
        
    def generate_layer_structure(self):
        """Generate the expected layer structure."""
        layers = []
        for i in range(32):  # 32 transformer layers
            layer_components = [
                f'layers.{i}.attention.wqkv',
                f'layers.{i}.attention.wo',
                f'layers.{i}.feed_forward.w1',
                f'layers.{i}.feed_forward.w2',
                f'layers.{i}.feed_forward.w3'
            ]
            layers.extend(layer_components)
        return layers

    def load_profile(self, file_path):
        """Load and parse Chrome trace format profile."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                events = data.get('traceEvents', [])
            else:
                events = data
                
            kernel_events = []
            for event in events:
                if (isinstance(event, dict) and 
                    event.get('ph') == 'X' and 
                    event.get('cat') == 'kernel'):
                    kernel_events.append({
                        'name': event['name'],
                        'duration': event['dur'],
                        'timestamp': event['ts'],
                        'args': event.get('args', {})
                    })
            
            print(f"Loaded {len(kernel_events)} kernel events from {file_path}")
            return kernel_events
        except Exception as e:
            print(f"Error loading profile {file_path}: {str(e)}")
            return []

    def match_kernel_to_component(self, kernel_name):
        """Match a kernel name to a layer component type using regex patterns."""
        for component, patterns in self.kernel_patterns.items():
            for pattern in patterns:
                if re.search(pattern, kernel_name, re.IGNORECASE):
                    if self.debug:
                        print(f"Matched kernel: {kernel_name[:100]}... to component: {component}")
                    return component
        if self.debug:
            print(f"No match found for kernel: {kernel_name[:100]}...")
        return None

    def analyze_profile(self, events):
        """Analyze kernel events and map to layer components."""
        layer_timings = defaultdict(float)
        layer_events = defaultdict(list)
        
        # First pass: group events by layer based on sequence
        current_layer = 0
        sequence_buffer = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        for event in sorted_events:
            component_type = self.match_kernel_to_component(event['name'])
            if component_type:
                sequence_buffer.append({
                    'event': event,
                    'component': component_type,
                    'duration': float(event['duration'])  # Convert to float explicitly
                })
                
                # When we have enough events for a complete layer
                if len(sequence_buffer) >= 3:  # Minimum events for a layer operation
                    layer_key_base = f'layers.{current_layer}'
                    
                    # Process the buffered events
                    for event_data in sequence_buffer:
                        layer_key = f"{layer_key_base}.{event_data['component']}"
                        layer_timings[layer_key] += event_data['duration']
                        layer_events[layer_key].append(event_data['event'])
                    
                    # Clear buffer and move to next layer if we see a pattern completion
                    if any(e['component'] == 'feed_forward.w3' for e in sequence_buffer):
                        sequence_buffer = []
                        current_layer += 1
                        if current_layer >= 32:
                            current_layer = 0
        
        if self.debug:
            print(f"\nProcessed {len(sorted_events)} events into {len(layer_timings)} layer components")
            total_time = sum(layer_timings.values())
            print(f"Total execution time: {total_time:.2f} μs")
            
            # Print some sample timings
            print("\nSample timings for first layer:")
            for component in ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                key = f'layers.0.{component}'
                if key in layer_timings:
                    print(f"{key}: {layer_timings[key]:.2f} μs")
                    
        return layer_timings

    def compare_profiles(self, int4_path, int8_path):
        """Compare INT4 and INT8 profiles with detailed layer matching."""
        int4_events = self.load_profile(int4_path)
        int8_events = self.load_profile(int8_path)
        
        int4_timings = self.analyze_profile(int4_events)
        int8_timings = self.analyze_profile(int8_events)
        
        # Create DataFrame for comparison
        layer_structure = self.generate_layer_structure()
        comparison_data = []
        
        for layer in layer_structure:
            comparison_data.append({
                'layer': layer,
                'int4_time': int4_timings.get(layer, 0),
                'int8_time': int8_timings.get(layer, 0),
                'difference': int8_timings.get(layer, 0) - int4_timings.get(layer, 0)
            })
        
        df = pd.DataFrame(comparison_data)
        return df

    def plot_comparisons(self, df):
        """Create visualizations for the comparison."""
        # Plot 1: Layer-wise comparison
        plt.figure(figsize=(20, 10))
        x = range(len(df))
        plt.bar(x, df['int4_time'], alpha=0.5, label='INT4')
        plt.bar(x, df['int8_time'], alpha=0.5, label='INT8')
        plt.xlabel('Layer Components')
        plt.ylabel('Duration (μs)')
        plt.title('INT4 vs INT8 Execution Time by Layer Component')
        plt.legend()
        plt.xticks(x[::5], df['layer'][::5], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('layer_comparison.png')
        plt.close()

        # Plot 2: Component type averages
        component_types = ['attention.wqkv', 'attention.wo', 
                         'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        
        avg_times = defaultdict(lambda: {'int4': 0, 'int8': 0})
        for _, row in df.iterrows():
            for comp_type in component_types:
                if comp_type in row['layer']:
                    avg_times[comp_type]['int4'] += row['int4_time']
                    avg_times[comp_type]['int8'] += row['int8_time']
        
        # Average by number of layers
        for comp_type in avg_times:
            avg_times[comp_type]['int4'] /= 32
            avg_times[comp_type]['int8'] /= 32

        plt.figure(figsize=(12, 6))
        x = range(len(component_types))
        width = 0.35
        plt.bar([i - width/2 for i in x], 
                [avg_times[ct]['int4'] for ct in component_types], 
                width, label='INT4')
        plt.bar([i + width/2 for i in x], 
                [avg_times[ct]['int8'] for ct in component_types], 
                width, label='INT8')
        plt.xlabel('Component Type')
        plt.ylabel('Average Duration (μs)')
        plt.title('Average Execution Time by Component Type')
        plt.xticks(x, component_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('component_comparison.png')
        plt.close()

def main():
    analyzer = TransformerProfileAnalyzer()
    
    try:
        # Compare profiles
        comparison_df = analyzer.compare_profiles(
            './profiles/int4_profile.json',
            './profiles/int8_profile.json'
        )
        
        # Save detailed comparison to CSV
        comparison_df.to_csv('layer_timing_comparison.csv', index=False)
        print("Detailed comparison saved to 'layer_timing_comparison.csv'")
        
        # Generate visualizations
        analyzer.plot_comparisons(comparison_df)
        print("Visualizations saved as 'layer_comparison.png' and 'component_comparison.png'")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Average INT4 execution time: {comparison_df['int4_time'].mean():.2f} μs")
        print(f"Average INT8 execution time: {comparison_df['int8_time'].mean():.2f} μs")
        print(f"Maximum time difference: {comparison_df['difference'].abs().max():.2f} μs")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()