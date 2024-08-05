#!/bin/bash
PID=$(ps -aux | grep uvicorn | grep -v grep | awk '{print $2}')
if [[ "$PID" == "" ]]; then
  echo "No uvicorn server found running."
else
  kill -2 $PID
  echo "Bot server stopped."
fi
echo