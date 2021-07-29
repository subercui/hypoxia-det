# Dependencies

```bash
pip install -r requirements.txt
```

# Training

First prepare the training images, run

```bash
cd data
python img_prep.py --source-folder path_to_source --target-folder path_to_target
cd -
```

To train the model, simply run

```bash
# request excution permission for the first time
chmod +x train.sh

./train.sh
```

Be aware that you can specific a image to plot during training by setting the `--test-img` argument in `train.sh`.

# Inference

We provide the `infer` method in the `train.py` for online inference. First provide images of RGB H&E slide, `necrosis.png` and `perfusion.png`; Then run the following command to predict the hypoxia output. The output will be saved as `predict.png`.

```bash
python train.py --infer \
        --patch-size 128 \
        --model path_to_checkpoint_file \
        --test-img path_to_source_images
```
