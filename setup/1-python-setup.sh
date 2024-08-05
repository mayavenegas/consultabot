#!/usr/bin/bash

source ../vars.sh

main() {
  # upgrade pip
  python3 -m pip install --upgrade pip
  build_llama_cpp
  download_model_from_hf
  install_python_libs
  clone_kb_repo
  echo
  echo
  echo "Log out and log back in again for correct env settings."
  echo
  echo
  echo
#  install_cuda_toolkit
}

#########################
install_cuda_toolkit() {
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda 
}

#########################
build_llama_cpp() {
  echo
  echo "Runtime ~20:00"
  time CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.2 -DCUDAToolkit_ROOT=/usr/local/cuda-12.2 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.2/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
}

#########################
download_model_from_hf() {
  #wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
  #wget https://huggingface.co/bartowski/Llama-3-Instruct-8B-SPPO-Iter3-GGUF/resolve/main/Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf

  model_url="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main"
  model_name=""Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
  if ! [ -f $MODEL_DIR/$model_name ]; then
    echo "Downloading LLM: $model_name..."
    wget $model_url/$model_name
    mkdir -p $MODEL_DIR
    mv $model_name $MODEL_DIR
  else
    echo "LLM exists in models directory - skipping download."
  fi

  git_lfs_version=$(git lfs version)
  if [[ "$git_lfs_version" != "git-lfs*"]]; then
    echo "Git large file support (lfs) not installed."
    exit -1
  fi
  model_url="https://huggingface.co/sentence-transformers"
  model_name="all-mpnet-base-v2"
  if ! [ -f $MODEL_DIR/$model_name ]; then
    echo "Cloning sentence-transformer model repo..."
    git clone $model_url/$model_name
    mv $model_name $MODEL_DIR
  else
    echo "Transformer exists in models directory - skipping download."
  fi
}

#########################
install_python_libs() {
  echo
  echo "Runtime ~2:20"
  echo
  time pip3 install pandas numpy torch scikit-learn joblib
  echo
  echo "Runtime ~20 seconds"
  echo
  time pip3 install sentence-transformers transformers bitsandbytes accelerate sentencepiece
  echo
  echo "Webscraping libs"
  time pip3 install BeautifulSoup4 selenium
  echo
  echo "bot-server libs"
  time pip3 install 			\
	  	langchain		\
	  	langchain_community     \
                langchain_huggingface   \
                faiss-gpu               \
                fastapi                 \
                uvicorn[standard]
  echo
  echo "bot-ui libs"
  time pip3 install streamlit
}

#########################
# clone repo & delete its .git dir to prevent updates
clone_kb_repo() {
  pushd ..
    git clone git@github.com:jodyhuntatx/cb-kb-docs.git
    cd cb-kb-docs
    rm -rf .git
  popd
}

main "$@"
