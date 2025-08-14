"""
Example usage script for the Play Dialogue Addressee Analyzer

This script demonstrates how to use the analyzer programmatically
and shows different configuration options.
"""

import os
import pandas as pd
from play_analyzer import initialize_model, process_dialogue_file

def create_sample_data():
    """Create a sample CSV file for testing."""
    sample_data = {
        'Speaker': ['HAMLET', 'HORATIO', 'HAMLET', 'HORATIO', 'HAMLET', 'MARCELLUS', 'HAMLET'],
        'Text': [
            'Who\'s there?',
            'Nay, answer me. Stand and unfold yourself.',
            'Long live the King!',
            'Bernardo?',
            'He.',
            'You come most carefully upon your hour.',
            '\'Tis now struck twelve. Get thee to bed, Francisco.'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_hamlet.csv', index=False)
    print("Created sample_hamlet.csv")
    return 'sample_hamlet.csv'

def basic_example():
    """Basic usage example."""
    print("=== Basic Example ===")
    
    # Create sample data
    input_file = create_sample_data()
    output_file = 'hamlet_analyzed_basic.csv'
    
    try:
        # Initialize model (this will download the model if not cached)
        print("Initializing model...")
        llm = initialize_model("meta-llama/Llama-3.1-8B-Instruct")
        
        # Process the file
        print("Processing dialogue...")
        process_dialogue_file(
            input_path=input_file,
            output_path=output_file,
            window_size=7,  # Process 7 lines at a time
            llm=llm,
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        )
        
        # Show results
        result_df = pd.read_csv(output_file)
        print("\nResults:")
        print(result_df[['Speaker', 'Text', result_df.columns[-1]]].to_string(index=False))
        
    except Exception as e:
        print(f"Error in basic example: {e}")

def advanced_example():
    """Advanced usage with different configurations."""
    print("\n=== Advanced Example ===")
    
    input_file = 'sample_hamlet.csv'
    
    # Different configurations to test
    configs = [
        {"window_size": 0, "output": "hamlet_whole_scene.csv"},  # Whole scene
        {"window_size": 3, "output": "hamlet_small_chunks.csv"},  # Small chunks
        {"window_size": 10, "output": "hamlet_large_chunks.csv"}  # Large chunks
    ]
    
    try:
        # Initialize model once
        llm = initialize_model("meta-llama/Llama-3.1-8B-Instruct")
        
        for config in configs:
            print(f"\nTesting window_size={config['window_size']}...")
            
            process_dialogue_file(
                input_path=input_file,
                output_path=config['output'],
                window_size=config['window_size'],
                llm=llm,
                model_name="meta-llama/Llama-3.1-8B-Instruct"
            )
            
            print(f"Results saved to {config['output']}")
    
    except Exception as e:
        print(f"Error in advanced example: {e}")

def batch_processing_example():
    """Example of processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    # This would be useful for processing multiple acts or plays
    input_files = ['sample_hamlet.csv']  # Add more files here
    output_dir = 'batch_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize model once for all files
        llm = initialize_model("meta-llama/Llama-3.1-8B-Instruct")
        
        for input_file in input_files:
            if os.path.exists(input_file):
                filename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, f"analyzed_{filename}")
                
                print(f"Processing {input_file}...")
                process_dialogue_file(
                    input_path=input_file,
                    output_path=output_file,
                    window_size=7,
                    llm=llm,
                    model_name="meta-llama/Llama-3.1-8B-Instruct"
                )
                
                print(f"Saved to {output_file}")
            else:
                print(f"File not found: {input_file}")
    
    except Exception as e:
        print(f"Error in batch processing: {e}")

def compare_models_example():
    """Example comparing different models."""
    print("\n=== Model Comparison Example ===")
    
    input_file = 'sample_hamlet.csv'
    models_to_test = [
        "meta-llama/Llama-3.1-8B-Instruct",
        # Add other models you want to compare
        # "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            
            # Initialize model
            llm = initialize_model(model_name)
            
            # Process with this model
            output_file = f"comparison_{model_name.split('/')[-1]}.csv"
            process_dialogue_file(
                input_path=input_file,
                output_path=output_file,
                window_size=7,
                llm=llm,
                model_name=model_name
            )
            
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

if __name__ == "__main__":
    print("Play Dialogue Addressee Analyzer - Examples")
    print("==========================================")
    
    # Run examples
    basic_example()
    advanced_example()
    batch_processing_example()
    
    # Uncomment to test model comparison (requires multiple models)
    # compare_models_example()
    
    print("\n=== All Examples Complete ===")
    print("Check the generated CSV files for results!")