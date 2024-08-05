#!/bin/bash
mkdir -p ./logs
tmux new -s bot-server uvicorn cb-no-state:app	\
        --host 0.0.0.0 --port 9000      	\
        --log-level info
