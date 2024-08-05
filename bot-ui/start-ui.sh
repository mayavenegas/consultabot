#!/bin/bash
mkdir -p ./logs
tmux new -s bot-ui streamlit run ./cb-ui.py   \
    --server.headless true              \
    --server.port 8000
