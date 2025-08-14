"""
Play Dialogue Addressee Analyzer

This script analyzes theatrical dialogue to identify who each character is addressing
in their lines. It uses a large language model to determine the receivers/addressees
for each line of dialogue in a play.

Requirements:
- vLLM library
- PyTorch with CUDA support (recommended)
- pandas
- Other dependencies listed in requirements.txt

Usage:
    python play_analyzer.py input_file.csv output_file.csv [--window-size 7] [--model MODEL_NAME]
"""

import os
import sys
import argparse
import json
import logging
import re
import time
from typing import List, Dict, Optional
import pandas as pd
import torch
import gc

# Import vLLM components
try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    print(f"Error: vLLM not available: {e}")
    print("Please install vLLM: pip install vllm")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# System prompt for the language model
SYSTEM_PROMPT = (
    "You are an expert in literature and addressee identification. "
    "You will receive lines from a play, with each line labeled by its speaker. "
    "Your job is to identify the receiver (target, addressees) for each line. "
    "Rules:\n"
    "- If the speaker is addressing all other characters, the receiver is 'ALL'\n"
    "- If the speaker talks to themselves, the receiver is the speaker's name\n"
    "- Return results in valid JSON format\n"
)


def check_system_resources():
    """Check and log available system resources."""
    try:
        import psutil
        logger.info(f"Available CPU cores: {psutil.cpu_count()}")
        logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    except ImportError:
        logger.info("psutil not available - skipping resource check")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {mem_total:.2f}GB total memory")
    else:
        logger.warning("CUDA not available - using CPU (not recommended for large models)")


def initialize_model(model_name: str = DEFAULT_MODEL) -> LLM:
    """
    Initialize the vLLM model with optimal settings.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Initialized LLM instance
    """
    check_system_resources()
    
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Determine optimal tensor parallel size based on available GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tensor_parallel_size = min(gpu_count, 8)  # Cap at 8 for most models
    
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            max_model_len=20000,  # Reasonable default
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        return llm
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise


def split_into_chunks(lines: List[str], window_size: int = 7) -> List[List[str]]:
    """
    Split lines into overlapping chunks for processing.
    
    Args:
        lines: List of dialogue lines
        window_size: Size of each chunk (0 means use all lines)
        
    Returns:
        List of chunks
    """
    if window_size <= 0 or len(lines) <= window_size:
        return [lines]
    
    chunks = []
    step = max(1, window_size - 2)  # Overlap for context
    
    for i in range(0, len(lines), step):
        chunk = lines[i:i + window_size]
        chunks.append(chunk)
    
    return chunks


def extract_json_from_response(content: str) -> Optional[Dict]:
    """
    Extract JSON data from model response using multiple strategies.
    
    Args:
        content: Raw model response
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(content.strip())
        if 'lines' in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON-like structures
    json_pattern = r'\{[^{}]*"lines"[^{}]*:\s*\[.*?\][^{}]*\}'
    matches = re.findall(json_pattern, content, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if 'lines' in parsed and isinstance(parsed['lines'], list):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Manual extraction
    lines_pattern = r'"line":\s*(\d+).*?"receivers":\s*\[(.*?)\]'
    matches = re.findall(lines_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        extracted_lines = []
        for line_num, receivers_str in matches:
            # Parse receivers
            receiver_matches = re.findall(r'"([^"]*)"', receivers_str)
            receivers = receiver_matches if receiver_matches else []
            
            extracted_lines.append({
                "line": int(line_num),
                "receivers": receivers
            })
        
        if extracted_lines:
            return {"lines": extracted_lines}
    
    logger.error("Failed to extract JSON from response")
    return None


def create_prompt(chunk: List[str], speakers: List[str]) -> List[Dict[str, str]]:
    """
    Create a formatted prompt for the model.
    
    Args:
        chunk: List of dialogue lines
        speakers: List of all speakers in the scene
        
    Returns:
        Formatted chat messages
    """
    user_input = (
        f"Characters in this scene: {', '.join(set(speakers))}\n\n"
        + "\n".join(chunk) + 
        "\n\nFor each line, identify who the speaker is addressing. "
        "Respond in this JSON format:\n"
        '{"lines": [{"line": 1, "speaker": "name", "receivers": ["receiver1"]}, ...]}\n\n'
        "Rules:\n"
        "- If addressing everyone: use ['ALL']\n"
        "- If talking to themselves: use [speaker_name]\n"
        "- Use exact line numbers shown\n"
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def process_dialogue_file(input_path: str, output_path: str, 
                         window_size: int = 7, llm: LLM = None, 
                         model_name: str = DEFAULT_MODEL) -> None:
    """
    Process a CSV file containing dialogue and identify addressees.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output CSV file
        window_size: Size of processing chunks (0 = whole scene)
        llm: Initialized LLM instance
        model_name: Name of the model for column naming
    """
    logger.info(f"Processing {input_path}")
    
    # Load dialogue data
    try:
        df = pd.read_csv(input_path)
        required_columns = ['Speaker', 'Text']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
            
        logger.info(f"Loaded {len(df)} lines of dialogue")
        
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        raise
    
    # Prepare data
    scene_lines = []
    speakers = []
    
    for i, row in df.iterrows():
        speaker = row['Speaker']
        text = str(row['Text'])
        scene_lines.append(f"Line {i+1}: {speaker}: {text}")
        speakers.append(speaker)
    
    logger.info(f"Unique speakers: {set(speakers)}")
    
    # Set up model parameters
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.1,
        top_p=0.9,
    )
    
    # Process chunks
    chunks = split_into_chunks(scene_lines, window_size)
    logger.info(f"Processing {len(chunks)} chunks")
    
    listeners_map = {}
    
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
        
        try:
            prompt = create_prompt(chunk, speakers)
            responses = llm.chat([prompt], sampling_params)
            
            if responses:
                content = responses[0].outputs[0].text
                parsed_data = extract_json_from_response(content)
                
                if parsed_data and 'lines' in parsed_data:
                    for line_data in parsed_data['lines']:
                        if 'line' in line_data and 'receivers' in line_data:
                            line_num = int(line_data['line'])
                            receivers = line_data['receivers']
                            
                            if isinstance(receivers, str):
                                receivers = [receivers]
                            
                            listeners_map[line_num] = receivers
                else:
                    logger.warning(f"Failed to parse response for chunk {chunk_idx + 1}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
    
    # Add results to dataframe
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    column_name = f'Receivers_{model_short_name}'
    
    df[column_name] = [
        ", ".join(listeners_map.get(i, ["Unknown"])) 
        for i in range(1, len(scene_lines) + 1)
    ]
    
    # Save results
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Report statistics
    unknown_count = sum(1 for i in range(1, len(scene_lines) + 1) if i not in listeners_map)
    success_rate = ((len(scene_lines) - unknown_count) / len(scene_lines)) * 100
    logger.info(f"Successfully processed {success_rate:.1f}% of lines")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Analyze play dialogue to identify addressees")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--window-size", type=int, default=7, 
                       help="Processing window size (0 for whole scene)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help="HuggingFace model name")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Initialize model
        logger.info("Initializing model...")
        llm = initialize_model(args.model)
        
        # Process file
        process_dialogue_file(
            args.input, 
            args.output, 
            args.window_size, 
            llm, 
            args.model
        )
        
        logger.info("Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()