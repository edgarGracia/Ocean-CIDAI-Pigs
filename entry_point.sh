echo "Cloning repository"
git clone https://github.com/edgarGracia/cidai_pigs.git

echo "Downloading model"
wget https://github.com/edgarGracia/cidai_pigs/releases/download/pigs_0_all_det2/pigs_0_all_model_final.pth -O model.pth
wget https://github.com/edgarGracia/cidai_pigs/releases/download/pigs_0_all_det2/pigs_0_all.yaml -O config.yaml

echo "Processing input path"
for d in /data/inputs/* ; do
  unzip -o "$d/0" -d input;
  python cidai_pigs/predict_path.py -c config.yaml -w model.pth -i input -o "output/$d" --use-cuda;
  rm -r input;
done

echo "Exporting output"
zip -r output.zip output
mv output.zip /data/outputs/