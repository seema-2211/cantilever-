# Advanced Image Captioning System

## Overview

This project generates natural language descriptions for images using:

- InceptionV3 for feature extraction
- LSTM decoder for caption generation
- Flickr8k dataset
- Flask web interface
- Docker deployment

## Features

- Automatic image caption generation
- Web-based image upload
- Deep learning architecture
- Dockerized deployment

## Dataset

Flickr8k Dataset

- 8091 images
- 5 captions per image

## Model Architecture

Image → InceptionV3 → Feature Vector

Caption Prefix → Embedding → LSTM

Feature + LSTM Output → Dense → Next Word

## Technologies

- Python
- TensorFlow
- Keras
- Flask
- Docker

## Run

```bash
python app.py
