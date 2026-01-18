#!/bin/bash
set -e

echo "📥 Downloading Cheff Models"

cd cheff
mkdir -p trained_models
cd trained_models

# Required models for full resolution generation
echo "Downloading autoencoder (~200MB)..."
[ ! -f "cheff_autoencoder.pt" ] && wget -q --show-progress https://syncandshare.lrz.de/getlink/fiQ6wTe7K7otQzyifNh9av/cheff_autoencoder.pt

echo "Downloading text-to-image model (~900MB)..."
[ ! -f "cheff_diff_t2i.pt" ] && wget -q --show-progress https://syncandshare.lrz.de/getlink/fi4R87B3cEWgSx4Wivyizb/cheff_diff_t2i.pt

echo "Downloading unconditional model (~900MB)..."
[ ! -f "cheff_diff_uncond.pt" ] && wget -q --show-progress https://syncandshare.lrz.de/getlink/fiE9pKbK38wzEvBrBCk95W/cheff_diff_uncond.pt

echo "Downloading super-resolution model (~400MB)..."
[ ! -f "cheff_sr_fine.pt" ] && wget -q --show-progress https://syncandshare.lrz.de/getlink/fiHM4uAfy7uxcfBXkefySJ/cheff_sr_fine.pt

echo ""
echo "✅ All models downloaded!"
echo "Location: $(pwd)"
ls -lh *.pt
