echo "Processing input path"
jq -r '.[]' <(echo $DIDS) | while read did; do
  unzip -o "/data/inputs/$did/0" -d /workdir/input;
  python /workdir/src/predict_path.py -c /workdir/model/config.yaml -w /workdir/model/model.pth -i /workdir/input -o "/workdir/output/$did" --use-cuda;
  python /workdir/src/NTracker/main.py --config-name coco_pigs images_path="/workdir/input" annotations_path="/workdir/output/$did/" hydra.run.dir="/workdir/output/$did/tracking/"
  rm -r /workdir/input;
done

echo "Exporting output"
zip -r output.zip /workdir/output/
mv output.zip /data/outputs/