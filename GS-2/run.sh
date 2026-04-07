#!/bin/bash

path_base=<dataset_base_path>
output_base=<output_base_path>

cuda_device=0
port=4060
dset=tandt

scenes=(
  "bicycle" "bonsai" "counter" "flowers" "garden"
  "stump" "treehill" "kitchen" "room"
  "Auditorium" "Ballroom" "Barn" "Caterpillar" "Church"
  "Courthouse" "Courtroom" "Family" "Francis" "Horse"
  "Ignatius" "Lighthouse" "M60" "Meetingroom" "Museum"
  "Palace" "Panther" "Playground" "Temple" "Train" "Truck"
)

factors=(4 2 2 4 4 4 4 2 2 $(for i in {1..21}; do echo 1; done))

for idx in "${!scenes[@]}"; do
  scene="${scenes[$idx]}"
  factor="${factors[$idx]}"

  path_output="$output_base/$dset/$scene"

  echo "Start: $scene"

  CUDA_VISIBLE_DEVICES="$cuda_device" python train.py \
    --port "$port" \
    --r "$factor" \
    -s "data/${scene}" \
    -m "$path_output" \
    --eval

  ((port++))
  sleep 5
done

echo "Done"