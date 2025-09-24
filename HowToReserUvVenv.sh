rm -rf .venv && uv venv --python 3.11

uv pip install -U pip wheel setuptools

uv pip install "numpy==1.25.2" "protobuf==4.25.3" "scipy==1.11.4" "pandas==2.2.2" "scikit-learn==1.3.2"

uv pip install "tensorflow==2.16.1" --no-cache-dir --force-reinstall

uv pip install -r requirements.txt --no-deps

