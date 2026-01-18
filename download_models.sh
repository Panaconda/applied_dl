#!/bin/bash
set -e

echo "📥 Downloading Cheff Models"

cd cheff
mkdir -p trained_models
cd trained_models

# Required models for full resolution generation
echo "Downloading autoencoder (~200MB)..."
if [ ! -f "cheff_autoencoder.pt" ]; then
    wget --show-progress --no-check-certificate -O cheff_autoencoder.pt.tmp https://syncandshare.lrz.de/getlink/fiQ6wTe7K7otQzyifNh9av/cheff_autoencoder.pt
    mv cheff_autoencoder.pt.tmp cheff_autoencoder.pt
fi

echo "Downloading text-to-image model (~900MB)..."
if [ ! -f "cheff_diff_t2i.pt" ]; then
    wget --show-progress --no-check-certificate -O cheff_diff_t2i.pt.tmp https://syncandshare.lrz.de/getlink/fi4R87B3cEWgSx4Wivyizb/cheff_diff_t2i.pt
    mv cheff_diff_t2i.pt.tmp cheff_diff_t2i.pt
fi

echo "Downloading unconditional model (~900MB)..."
if [ ! -f "cheff_diff_uncond.pt" ]; then
    wget --show-progress --no-check-certificate -O cheff_diff_uncond.pt.tmp https://syncandshare.lrz.de/getlink/fiE9pKbK38wzEvBrBCk95W/cheff_diff_uncond.pt
    mv cheff_diff_uncond.pt.tmp cheff_diff_uncond.pt
fi

echo "Downloading super-resolution model (~400MB)..."
if [ ! -f "cheff_sr_fine.pt" ]; then
    wget --show-progress --no-check-certificate -O cheff_sr_fine.pt.tmp https://syncandshare.lrz.de/getlink/fiHM4uAfy7uxcfBXkefySJ/cheff_sr_fine.pt
    mv cheff_sr_fine.pt.tmp cheff_sr_fine.pt
fi

echo ""
echo "✅ All models downloaded!"
echo "Location: $(pwd)"
ls -lh *.pt
