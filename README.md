# Animal Sound Classifier with PyTorch

This project is a neural network classifier for animal sounds, using PyTorch with Metal Performance Shaders (MPS) support for Apple Silicon. The classifier can predict the valence and context of animal sounds from spectrogram images.

## Features

- ResNet50-based neural network classifiers for both valence and context
- Support for Apple Silicon GPUs via Metal Performance Shaders (MPS)
- Web interface for uploading and analyzing audio files
- Support for spectrogram generation from audio files

## Requirements

- Python 3.12 or higher
- Apple Silicon Mac (M1/M2/M3) for MPS acceleration
- UV package manager

## Installation

1. Install UV if you don't have it already:
   ```
   curl -sSf https://install.ultraviolet.rs | bash
   ```

2. Clone the repository:
   ```
   git clone https://github.com/yourusername/c4c-animal-sounds.git
   cd c4c-animal-sounds
   ```

3. Create a virtual environment and install dependencies:
   ```
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

## Usage

### Training Models

To train the neural network models:

```
uv run overall_nns.py
```

This will train both valence and context models and save checkpoints to the `checkpoints` directory.

### Making Predictions

To make predictions on test data:

```
uv run predict.py
```

### Running the Web Server

To run the Flask web server:

```
uv run server.py
```

Then open a browser and navigate to `http://127.0.0.1:5000`.

## Model Checkpoint Files

The trained models are saved as `.pt` files in the following locations:
- Valence model: `checkpoints/Valence/Iter1/model_checkpoint_best.pt`
- Context model: `checkpoints/Context/Iter1/model_checkpoint_best.pt`

## Project Structure

- `overall_nns.py`: Main script for training neural networks
- `nn_function.py`: Core neural network functionality
- `nn_prep_for_classify.py`: Prepares neural networks for classification
- `predict.py`: Script for making predictions on new data
- `server.py`: Flask web server for the user interface
- `site_validation_imds.py`: Dataset utilities for validation
- `metrics.py`: Functions for calculating performance metrics
- `spectrogram_maker.py`: Generates spectrograms from audio files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet50 architecture is used from the PyTorch model zoo
- The original MATLAB implementation that this project was based on

# Summary
This project was developed as part of Code 4 Compassion Hackathon 2025. The goal was to validate and extend the implementation from this [paper](https://www.nature.com/articles/s41598-022-07174-8). The original authors sought to develop a model that could classify the emotional valence of pigs throughout all stages of life. Although the original code was written in MATLAB, we aimed to refactor the original code in Python to make it more accessible and portable. A quick web interface was put together to demonstrate one implementation of the user workflow.

In a future, more mature implementation, the application would be able to access live audio recordings (e.g., from a mobile device), process the streamed data, and generate predictions of the valence of the pig calls in real-time. As an example, we can envision this application being useful in cases where factory farm practices must be validated and/or monitored and held accountable.


# Dataset
https://zenodo.org/records/8252482


# Example of the Interface
![image](https://github.com/user-attachments/assets/c3684deb-5c41-4c67-99cf-a23ced50070b)
