
#!/bin/bash

# isort
isort .

## emacs noise ;)
find . -name '*~' -type f -delete

## other noise - more robust cleanup
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

## import cleanup
find . -name '*.py' | xargs autoflake --in-place --remove-unused-variables --expand-star-imports

## formatting
find . -name '*.py' -print0 | xargs -0 yapf -i

flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
flake8 . --count --exit-zero --max-complexity=10 --statistics
