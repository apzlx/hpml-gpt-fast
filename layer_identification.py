import re

def parse_layer_results(log_content):
    """
    Parse layer results from log content and extract layer name, accuracy, and improvement.
    
    Args:
        log_content (str): Content of the log file
        
    Returns:
        list: List of tuples containing (layer_name, accuracy, improvement)
    """
    results = []
    
    # Pattern to match the complete result block
    pattern = r"Results for layer: (layers\.[^\n]+)\nAccuracy: (\d+\.\d+)\nImprovement over INT4 baseline: (-?\d+\.\d+)"
    
    # Find all matches
    matches = re.finditer(pattern, log_content, re.MULTILINE)
    
    # Extract information from each match
    for match in matches:
        layer_name = match.group(1)
        accuracy = float(match.group(2))
        improvement = float(match.group(3))
        results.append((layer_name, accuracy, improvement))
    
    return results

def main():
    try:
        # Read the log file
        with open('layer_analysis.log', 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Parse results
        results = parse_layer_results(log_content)
        
        # Print results in a formatted way
        print("\nLayer Analysis Results:")
        print("-" * 70)
        print(f"{'Layer':<40} {'Accuracy':<10} {'Improvement':<10}")
        print("-" * 70)
        for layer, accuracy, improvement in results:
            print(f"{layer:<40} {accuracy:>8.1f}%  {improvement:>+9.1f}")
            
    except FileNotFoundError:
        print(f"Error: Could not find log file 'layer_analysis.log'")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()