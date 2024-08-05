#!/bin/bash
PS_OUT=$(ps -aux | grep uvicorn | grep -v grep)
if [[ "$PS_OUT" == "" ]]; then
  echo "No uvicorn server found running."
else
  echo $PS_OUT
fi
echo