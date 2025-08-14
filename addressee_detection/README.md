# Play Dialogue Addressee Analyzer

A tool that uses large language models to analyze theatrical dialogue and identify who each character is addressing in their lines.

## Features

- Analyzes CSV files containing play dialogue
- Identifies addressees (receivers) for each line of dialogue
- Uses state-of-the-art language models via vLLM for fast inference
- Supports chunked processing for long scenes
- Configurable window sizes for context management

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for large models)
- At least 16GB RAM (more recommended for larger models)

### Setup

1. Clone this repository:
```bash
git clone [your-repo-url]
cd play-dialogue-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For CUDA support, install PyTorch with CUDA:
```bash
# Visit https://pytorch.org/get-started/locally/ for the latest command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Input Format

Your CSV file should contain at least these columns:
- `Speaker`: The character name
- `Text`: The dialogue text

Example:
```csv
Speaker,Text
HAMLET,To be or not to be, that is the question
OPHELIA,My lord, how fares your disposition?
HAMLET,I humbly thank you; well, well, well.
```

### Basic Usage

```bash
python play_analyzer.py input.csv output.csv
```

### Advanced Usage

```bash
# Use a different model
python play_analyzer.py input.csv output.csv --model meta-llama/Llama-3.1-8B-Instruct

# Process with different window size
python play_analyzer.py input.csv output.csv --window-size 10

# Process entire scene at once (no chunking)
python play_analyzer.py input.csv output.csv --window-size 0
```

### Parameters

- `--window-size`: Number of lines to process together (default: 7)
  - Use 0 to process the entire scene at once
  - Larger windows provide more context but use more memory
- `--model`: HuggingFace model name (default: meta-llama/Llama-3.3-70B-Instruct)

## Output Format

The tool adds a new column to your CSV with the identified receivers for each line:

```csv
Speaker,Text,Receivers_Llama-3.3-70B-Instruct
HAMLET,To be or not to be...,HAMLET
OPHELIA,My lord how fares...,HAMLET
HAMLET,I humbly thank you...,OPHELIA
```

### Receiver Types

- **Character names**: When addressing specific characters
- **ALL**: When addressing all characters present
- **Speaker's name**: When talking to themselves (soliloquy)
- **Unknown**: When the model couldn't determine the addressee

## Model Support

The tool supports any instruction-tuned language model available through vLLM, including:

- Llama 3.1/3.3 series
- Mistral models
- CodeLlama models
- Other compatible models

### Hardware Requirements

| Model Size | Recommended GPU Memory | Tensor Parallel Size |
|------------|------------------------|---------------------|
| 7B-8B      | 16GB+                 | 1                   |
| 13B        | 24GB+                 | 1-2                 |
| 70B        | 80GB+ (A100)          | 4-8                 |

## Examples

### Processing Shakespeare

```bash
python play_analyzer.py hamlet_act1.csv hamlet_analyzed.csv --window-size 5
```

### Processing Modern Drama

```bash
python play_analyzer.py death_of_salesman.csv analyzed_output.csv --window-size 10
```

### Using Smaller Model

```bash
python play_analyzer.py play.csv output.csv --model meta-llama/Llama-3.1-8B-Instruct
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce window size or use a smaller model
2. **Model Loading Fails**: Check if you have sufficient GPU memory
3. **No CUDA Available**: The tool will work on CPU but will be much slower

### Performance Tips

1. Use GPU with sufficient memory for your chosen model
2. Adjust window size based on scene length and complexity
3. For very long plays, consider processing acts separately
4. Monitor GPU memory usage during processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your chosen license here]

## Citation

If you use this tool in academic research, please cite:

```
[Add citation format here]
```

## Support

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide system information and error logs when reporting bugs