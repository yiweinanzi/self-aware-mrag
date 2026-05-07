#!/bin/bash

# Set up working directories
echo "Setting up directories..."
mkdir -p ./data
cd ./data || exit

# Configure HuggingFace mirror for faster downloads in China
echo "Configuring HuggingFace mirror..."
export HF_ENDPOINT=https://hf-mirror.com

# Download ViDoSeek dataset
echo "Downloading ViDoSeek dataset..."
huggingface-cli download autumncc/ViDoSeek --repo-type dataset --local-dir .

# Create directories for organizing data
echo "Creating directories for organizing data..."
mkdir -p ViDoSeek
mkdir -p SlideVQA

# Extract videoseek_pdf_document.zip
echo "Extracting videoseek_pdf_document.zip..."
unzip -o vidoseek_pdf_document.zip -d ViDoSeek/

# Extract slidevqa_pdf_document.zip
echo "Extracting slidevqa_pdf_document.zip..."
unzip -o slidevqa_pdf_document.zip -d SlideVQA/

echo "Download and extraction complete!"
echo "Files organized in data/ViDoSeek and data/SlideVQA folders"

cd ..
