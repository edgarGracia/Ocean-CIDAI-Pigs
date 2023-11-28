jq -r '.[]' <(echo $DIDS) | while read did; do
  unzip -o "/data/inputs/$did/0" -d /workdir/input &>> logs;
  python /workdir/src/predict_path.py -c /workdir/model/config.yaml -w /workdir/model/model.pth -i /workdir/input -o "/workdir/output/$did" --use-cuda &>> logs;
  python /workdir/src/NTracker/main.py --config-name coco_pigs images_path="/workdir/input" annotations_path="/workdir/output/$did/" hydra.run.dir="/workdir/output/$did/tracking/" &>> logs;
  rm -r /workdir/input;
done

zip -r output.zip /workdir/output/ &>> logs
sudo mv output.zip /data/outputs/
sudo mv logs /data/outputs/out_logs