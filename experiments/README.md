## Configuration files

These are the configuration files for the networks described in the paper.

In section [model], we specify hyperparameters of the trained model.

| Parameter | Default | Description |
| -------- | ------- | ------- |
| type | ASRSpecNet | Model class |
| n_mel | 128 | Number of mel frequencies |
| n_mfcc_out | 128 | [MFCCs: unused] |
| cnn_out_size | 128 | Number of filters in the last CNN layer |
| rnn_hidden_size | 256 | Hidden size of the RNN layers |
| num_rnn_layers | 1 | Number of RNN layers within a module (Phoneme module & Word module) |
| rnn_bidirectional | False | Use bidirectional RNNs |
| rnn_drop | 0.5 | Dropout rate used during trainig |
| downsample_factor | 160 | Hop length during spectrogram computation |
| vocabulary_size | 4000 | [Word vocabulary size: unused/fixed] |

In section [pretraining], we specify the training data as well as hyperparameters of the training procedure.

| Parameter | Default | Description |
| -------- | ------- | ------- |
| pretraining_manifest_train | | Text file listing the training files (has to be placed in `datapath/`) |
| pretraining_manifest_dev | | Text file listing the validation files |
| pretraining_manifest_test | | Text file listing the test files |
| fs | 16000 | [unused] |
| time_shift | 0 | [unused] |
| n_output_quantize | 0 | [unused] |
| pretraining_lr | 0.001 | Initial learning rate during training |
| pretraining_patience | 5 | Patience of the learning rate scheduler (number of epochs) |
| pretraining_lr_factor | .1 | Factor to decrease learning rate after no improvement has been observed for {pretraining_patience} epochs |
| pretraining_batch_size | 64 | Batch size during training |
| pretraining_num_epochs | 30 | Number of epochs |
| pretraining_length_mean | 2.25 | Average length of the sampled training clips |
| pretraining_length_var | 1 | Variance of the length distribution to sample training clips |
| grad_clip | 1.0 | Gradient clipping threshold |
| pretraining_phoneout | 1 | Train a phoneme classifier (this is e.g. 0 for the word-only model) |
| pretraining_wordout | 1 | Train a word classifier |
| pretraining_langin | 0 | Use language tags (0: no language tags, 1: use language tags half of the time, 2: use language tags) |
| pretraining_langin_test | 0 | Use language tags during testing |
| pretraining_p_noise | 0.0 | [unused] |

The section [training] is only relevant in some instances: Here, the main network, that was trained during "pretraining", is frozen and another network is trained on top of it. This was only used for the alternative models shown in Figure S4 (Homologs & Words, Word-only), for which we trained a phoneme classifier on top of the pretrained network.

The hyperparameter are analogous to section [pretraining]

| Parameter | Default | Description |
| -------- | ------- | ------- |
| training_manifest_train |  | Text file listing the training files |
| training_manifest_dev |  | Text file listing the validation files |
| training_manifest_test |  | Text file listing the test files |
| training_type | ASRSpecNetPhonemes | Model class to train on top of the pretrained one |
| training_lr | 0.001 | Initial learning rate |
| training_patience | 5 | Patience of the learning rate scheduler |
| training_batch_size | 64 | Batch size |
| training_num_epochs | 1 | Number of epochs |
