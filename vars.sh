export PATH=$PATH:/home/ubuntu/.local/bin
export HUGGINGFACE_TOKEN=foo

# Prompt for HF token if not already set
if [[ "$HUGGINGFACE_TOKEN" == "" ]]; then
  echo -n "Please enter your HuggingFace API token: "
  unset HUGGINGFACE_TOKEN
  read token
  export HUGGINGFACE_TOKEN=$token
fi

export USER_AGENT='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 OPR/45.0.2552.888'
export MODEL_DIR=/home/ubuntu/models
