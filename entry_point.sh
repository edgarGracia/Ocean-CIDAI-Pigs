echo "Processing input path"
jq -r '.[]' <(echo $DIDS) | while read did; do
  unzip -o "/data/inputs/$did/0" -d /workdir/input;
  yolo segment predict model=/workdir/model/model.pt source=/workdir/input save=True save_txt=True project=/workdir/output/ name="$did" retina_masks=True;
  rm -r /workdir/input;
done

echo "Exporting output"
zip -r output.zip /workdir/output/
mv output.zip /data/outputs/