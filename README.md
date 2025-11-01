# Jumbled Frame Reconstruction

## Overview
This project reconstructs a shuffled 10-second, 1080p, 30 FPS video by reordering its frames into the correct sequence. It uses ORB feature detection, similarity matrix computation with multiprocessing acceleration, and Traveling Salesman Problem (TSP) optimization for frame ordering.

## Motivation
Reordering shuffled frames is a challenging task with applications in video forensics, restoration, and analysis. This project provides an efficient approach to recover the original video sequence automatically.

## Features
- Frame feature extraction using ORB for fast, reliable matching.
- Multiprocessing to speed up pairwise similarity calculation.
- Traveling Salesman Problem formulation for optimal frame ordering.
- Automatic correction of playback direction for natural video flow.
- Adjustable frame count and resolution for performance tuning.

## Technologies Used
- Python 3.12
- OpenCV (opencv-contrib-python)
- NetworkX for graph and TSP algorithms
- TQDM for progress visualization
- Multiprocessing module

## Installation

1. Clone or download this repository.
2. Create and activate a Python virtual environment:

   On Windows:
python -m venv videoenv

3. Install dependencies:
pip install -r requirements.txt
## Usage

1. Extract frames from the jumbled video:
python extractframes.py


2. Run the reconstruction to reorder frames into a video:
python reconstruct.py


3. Adjust parameters in `reconstruct.py` as needed:

- `max_frames` to control number of frames processed (default 300).
- `scale` in `preprocess_frame()` for quality vs. speed tradeoff.

## Output

- The output reconstructed video is saved as `reconstructed.mp4` at 30 FPS.
- The playback direction is automatically handled; reversed playback is the natural and correct one for this project.
- No manual reversal of output video is necessary.

## Algorithm Summary

- Frames are resized and ORB features extracted.
- Pairwise frame similarity is computed in parallel.
- TSP approximates the best order minimizing frame dissimilarity.
- Playback direction is chosen based on similarity scores.
- Reordered frames compiled into final video.

## Performance Notes

- Processing all 300 frames at full resolution takes time.
- Frame resizing and multiprocessing speed up the pipeline.
- It’s recommended to test on fewer frames before full reconstruction.

## Troubleshooting

- Slow processing? Lower `max_frames` or reduce `scale`.
- Missing dependencies? Activate virtual environment and run `pip install -r requirements.txt`.
- Use `.gitignore` to avoid committing large folders/files like `frames/` or videos.

## License

MIT License

## Contact

Your Name - Gun sharma;
Email - sharmagun2904@gmail.com;

---

Thank you for using Jumbled Frame Reconstruction!


