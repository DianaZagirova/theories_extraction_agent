#!/bin/bash
# Run mechanism extraction test with automatic confirmation

echo "Running mechanism extraction test on 15 theories..."
echo "y" | python test_mechanism_small_sample.py --sample-size 15
