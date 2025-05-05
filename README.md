# Code repository for: "Efficient neural encoding as revealed by bilingualism"

<!---
<img src="https://pwdonh.github.io/images/bilingual.png" alt="" height="250">
-->

## Paper reference

Charlotte Moore*, Peter Donhauser*, Denise Klein, Krista Byers-Heinlein, "Efficient neural encoding as revealed by bilingualism." PsyArxiv, 2025. [https://osf.io/preprints/psyarxiv/m8vdj_v1](https://osf.io/preprints/psyarxiv/m8vdj_v1)

## Before starting

* This code was tested on a Linux machine using Python 3.9
* Training models will require a GPU, but working with the pretrained models doesn't
* Before running the code, make sure you first install the required packages
    ```
    pip install -r requirements.txt
    ```
* Then install the audio functions in `torchaudio-contrib`
    ```
    cd torchaudio-contrib
    pip install .
    cd ..
    ```

## Training

You can skip this if you want to work with the pretrained models.

To train a model we specify hyperparameters and training data in a configuration file. The ones that were used in the paper can be found under [experiments/](experiments/). There you will also find a description of all the hyperparameters. 

Model training is then started using the `main.py` script. For this you have to first download the training data as described in [data/](data/). To train the bilingual French-English model, run:

```
python main.py --config_path experiments/commonvoice_fren_phonemes_words.cfg --datapath ./data/ --pretrain
```

The trained model parameters will be save in a directory named after the configuration file [experiments/commonvoice_fren_phonemes_words/pretraining/](experiments/commonvoice_fren_phonemes_words/pretraining/)

To evaluate frame-by-frame accuracy of the trained model separately for the two languages, you can run:

```
python proc_3_accuracy_per_language.py --config_path experiments/commonvoice_fren_phonemes_words.cfg
```

which will save the results in [experiments/commonvoice_fren_phonemes_words/pretraining/accuracy_per_language.csv](experiments/commonvoice_fren_phonemes_words/pretraining/accuracy_per_language.csv)

## Example sentences

This repository includes two examples sentences generated with Google text-to-speech.

The notebook [example_sentences.ipynb](example_sentences.ipynb) shows how to load these audio files, run them through a pretrained network and inspect the output.

## Results from the paper

The following scripts can be used to reproduce the results from the paper

* Re-training networks (takes a lot of time, you can skip this and use pretrained models)
  ```
  for lang_code in {en,fr,de,fren,defr,deen,defren}
  do
    python main.py --config_path experiments/commonvoice_${lang_code}_phonemes_words.cfg --pretrain
  done
  ```
* Figure 1: hidden activity traces, 3d plots
  ```
  python proc_1_representation.py
  python proc_2_example_sentences.py
  python figure_1_traces.py
  ```
* Figure S1: example sentence, demonstration of frame-by-frame accuracy computation
  ```
  python figure_s1_accuracy_method.py
  ```
* Figure 2: 3d phoneme representations for English, French-English & French-English-German networks
  ```
  python proc_1_representation.py
  python figure_2_representation.py
  ```
* Figure S2: feature decodability in fully trained networks
  ```
  python proc_4_decodability.py
  python figure_s2_decodability.py
  ```
* Figure 3: development of representations, simultaneous vs. sequential bilingual networks
  ```
  python proc_5_sequential.py --step train --run 1
  python proc_5_sequential.py --step decodability --run 1
  # repeat for run 2, 3, 4, 5, 6, then:
  python figure_3_decodability.py
  ```
* Figure 4: accuracy development, simultaneous vs. sequential bilingual networks
  ```
  python proc_5_sequential.py --step accuracy --run 1
  # repeat for run 2, 3, 4, 5, 6, then:
  python figure_4_accuracy.py
  ```
* Figure S4: alternative models, "Homologs & words` as well as "Word only"
  ```
  # the "--train" flag ensures that after pretraining the main network is frozen and a phoneme classifier is trained on top of it
  python main.py --config_path experiments/commonvoice_fren_homologes_words.cfg --pretrain --train
  python main.py --config_path experiments/commonvoice_fren_words.cfg --pretrain --train
  python proc_4_decodability.py
  python figure_s4_alternative_models.py
  ```

## This repository

The code base for this project was started as a fork of [this repository](https://github.com/lorenlugosch/end-to-end-SLU) and modified thereafter. Here is the corresponding paper:

Lugosch, L., Ravanelli, M., Ignoto, P., Tomar, V. S., & Bengio, Y. (2019). Speech model pre-training for end-to-end spoken language understanding. arXiv preprint arXiv:1904.03670.
