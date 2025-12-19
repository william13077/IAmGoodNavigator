#!/bin/bash

# Create demo directory if it doesn't exist
mkdir -p demo

# Navigate into demo directory
cd demo

# Download demo_scene.zip
echo "Downloading demo_scene.zip..."
wget -L -O demo_scene.zip "https://www.dropbox.com/scl/fi/5q23dpt8l1irxoi892njg/demo_scene.zip?rlkey=oxi70q0sac65wfs4e6bntfakm&st=xz5txoq0&dl=1"

# Unzip the files
echo "Unzipping files..."
unzip -o demo_scene.zip

# Optional: clean up zip files
rm demo_scene.zip

echo "Done!"