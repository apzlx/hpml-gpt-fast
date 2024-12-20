import re

def parse_layer_results(log_content):
    results = []
    pattern = r"Results for layer: (layers\.[^\n]+)\nAccuracy: (\d+\.\d+)\nImprovement over INT4 baseline: (-?\d+\.\d+)"
    matches = re.finditer(pattern, log_content, re.MULTILINE)
    for match in matches:
        layer_name = match.group(1)
        accuracy = float(match.group(2))
        improvement = float(match.group(3))
        results.append((layer_name, accuracy, improvement))
    
    return results

def main():
    try:
        with open('layer_analysis.log', 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        results = parse_layer_results(log_content)
        
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