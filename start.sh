#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the build directory
BUILD_DIR="csnn-simulator-build"

# Remove the existing build directory if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create a new build directory
echo "Creating build directory..."
mkdir "$BUILD_DIR"

# Change into the build directory
cd "$BUILD_DIR"

# Run CMake with specified options
echo "Running CMake..."
cmake ../ -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO

# Compile the project
echo "Building the project..."
make

echo "Build completed successfully."

