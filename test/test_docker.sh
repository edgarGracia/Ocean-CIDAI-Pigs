docker run --rm \
-v "$(pwd)/data/inputs":/data/inputs \
-v "$(pwd)/data/outputs":/data/outputs \
-v "$(pwd)/../entry_point.sh":/data/transformations/algorithm \
-e DIDS='["1234"]' \
egracia/cidai_pigs:pigs_0_all_det2 bash /data/transformations/algorithm