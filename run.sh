#!/bin/bash

source .venvLinux/bin/activate
start=$(date +%s)
python src/Models/Person_model_saves.py
end=$(date +%s)
echo "Elapsed $(($end - $start)) seconds"
