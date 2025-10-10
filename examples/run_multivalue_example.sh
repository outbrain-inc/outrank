#!/bin/bash

# End-to-end example for multivalue features with OutRank
# This script demonstrates the complete workflow

echo "========================================================================"
echo "OutRank Multivalue Features - End-to-End Example"
echo "========================================================================"
echo ""
echo "This example demonstrates:"
echo "  1. Using '_' delimiter for multivalue features (not ',')"
echo "  2. Direct multivalue MI computation (no expansion needed)"
echo "  3. Integration with OutRank's feature ranking API"
echo ""

# Run the end-to-end Python example
python3 examples/multivalue_end_to_end.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Example completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  - See examples/multivalue_data.csv for sample data format"
    echo "  - Use '_' as delimiter in your CSV files for multivalue features"
    echo "  - Use heuristics: MI-multivalue-jaccard, MI-multivalue-overlap, MI-multivalue-set"
    echo ""
else
    echo ""
    echo "Example failed! Check the error messages above."
    exit 1
fi
