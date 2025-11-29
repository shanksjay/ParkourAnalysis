#!/bin/bash

# Parkour Analysis Setup Script
# Sets up the environment using UV (recommended) or pip

set -e

echo "🏃 Setting up Parkour Analysis System..."

# Check if UV is installed
if command -v uv &> /dev/null; then
    echo "✅ UV detected - using UV for setup"
    USE_UV=true
else
    echo "⚠️  UV not found - using pip instead"
    echo "   Install UV for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if [ "$USE_UV" = true ]; then
    # UV setup
    echo "📦 Creating UV virtual environment..."
    uv venv
    
    echo "🔌 Activating virtual environment..."
    source .venv/bin/activate
    
    echo "📥 Installing dependencies with UV..."
    # Install pip first (required for chumpy build)
    echo "   Installing pip (required for chumpy)..."
    uv pip install pip
    # Then install all dependencies
    uv pip install -r requirements.txt
    
    echo "✅ UV setup complete!"
else
    # pip setup
    echo "📦 Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ Virtual environment created"
    else
        echo "ℹ️  Virtual environment already exists"
    fi
    
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    
    echo "✅ pip setup complete!"
fi

echo ""
echo "📝 Next steps:"
echo "1. Download SMPL-X models from: https://smpl-x.is.tue.mpg.de/"
echo "2. Place your parkour video as 'input_parkour.mp4'"
echo "3. Run: python parkour_analysis.py"
echo ""
echo "💡 Note: YOLO model will auto-download on first run"
echo ""

