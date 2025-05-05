## Example sentences

The data included in this repository are two speech samples generated using Google text-to-speech of the sentences: "I grew up with two languages" and "J'ai grandi avec deux langues."

## Full training dataset

We used the [Common Voice dataset](https://commonvoice.mozilla.org/) by the Mozilla foundation.

For this paper, we renamed the individual files including the speaker and sentence IDs. We also created a phoneme- and word-level alignment using the [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).

These can be downloaded from the following link: [does not exist yet]

Extract the tarballs in this directory
```
tar -xvzf en_archive.tar.gz
tar -xvzf fr_archive.tar.gz
tar -xvzf de_archive.tar.gz
```
These include the wav-files and TextGrid alignments, as well as the text files specifying our training/validation/test splits.
