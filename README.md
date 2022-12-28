# Music Separated Speech Enhancement using Image Translation 
 
# Table of contents

* [Introduction](#introduction)
* [Quick Reference](#quick-reference)
* [Dataset](#dataset)
* [Method](#method)
* [Training](#training)
* [Results](#results)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)
* [References](#references)

# Introduction
>[Table of contents](#table-of-contents)

Source separation can be defined as the process of separating a set of source signals from a set of mixed signals. Source separation for music is the process of separating music multiple sources of music, which can be vocals, drums, bass, etc. 'Music Separated Speech' is speech that has been separated from music. When speech is separated from background music, it usually contains certain artifacts. 

In this project, we first create a dataset using a subject of the LJ Dataset. We add music to the recordings and pass them through spleeter which is a music source separator. We then use apply image translation to mel-spectrograms where we predict a clean mel-spectrogram from that of a separated one. Once that's done, we use the MelGAN vocoder, which has been trained on the LJ dataset, to generate raw audio.

# Quick reference
>[Table of contents](#table-of-contents)

### Environment setup
Clone this repository to your system.
```
$ git clone https://github.com/hmartelb/Pix2Pix-Timbre-Transfer.git
```
Make sure that you have Python 3 installed in your system. It is recommended to create a virtual environment to install the dependencies. Open a new terminal in the master directory and install the dependencies from requirements.txt by executing this command:
```
$ pip install -r requirements.txt
```
### Dataset generation
Download the NSynth Dataset and the Classical Music MIDI Dataset.
* The NSynth Dataset, “A large-scale and high-quality dataset of annotated musical notes.” 
https://magenta.tensorflow.org/datasets/nsynth

* Classical Music MIDI Dataset, from Kaggle 
https://www.kaggle.com/soumikrakshit/classical-music-midi

Generate the audios and the features with the following scripts. Optional arguments are displayed in brackets “[ ]”.
```
$ python synthesize_audios.py --nsynth_path <NSYNTH_PATH>
                              --midi_path <MIDI_PATH>
                              --audios_path <AUDIOS_PATH>
                             [--playback_speed <PLAYBACK_SPEED>]
                             [--duration_rate <DURATION_RATE>]
                             [--transpose <TRANSPOSE>]
```

```
$ python compute_features.py --audios_path <AUDIOS_PATH> 
                             --features_path <FEATURES_PATH>
```
### Pix2Pix training
Train the Pix2Pix network with the ``train.py`` script, specifying the instrument pair to convert from origin to target, and the path where the dataset is located. 
```
$ python train.py --dataset_path <DATASET_PATH> 
                  --origin <ORIGIN>
                  --target <TARGET>
                 [--gpu <GPU>] 
                 [--epochs <EPOCHS>]
                 [--epoch_offset <EPOCH_OFFSET>] 
                 [--batch_size <BATCH_SIZE>]
                 [--gen_lr <GENERATOR_LEARNING_RATE>] 
                 [--disc_lr <DISCRIMINATOR_LEARNING_RATE>]
                 [--validation_split <VALIDATION_SPLIT>] 
                 [--findlr <FINDLR>]
```

# Dataset
>[Table of contents](#table-of-contents)

The dataset has been creating using a subset for the LJ dataset and can be downloaded here. 

takes in audio files and creates a dataset
the dataset contains '.pickle' files where each
file contains an input and target mel-spectrogram 

# Method
>[Table of contents](#table-of-contents)

The Pix2Pix architecture has been designed for image processing tasks, but in this case the format of the data is audio. Therefore, a preprocessing step to convert a 1D signal (audio) into a 2D signal (image) is required.

# Training
>[Table of contents](#table-of-contents)

The adversarial networks have been trained in a single GTX 1080Ti GPU for 100 epochs using magnitude spectrograms of dimensions (256,256,1), a validation split of 0.1, 22875 examples per instrument pair, Adam optimizer, and Lambda of 100 as in the original Pix2Pix paper. 

### Batch size

After some inconclusive experiments setting the batch size to 1, 2 and 4, the best convergence has been achieved using a batch size of 8. This gives a total of 2859 iterations per epoch.
>In the case of the conditioned model the number of training examples is 68625, which gives 8578 iterations per epoch.

### Learning rate

The learning rate has been searched using the Learning Rate Finder method mentioned in [this blog post from Towards Data Science](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) and [this paper](https://arxiv.org/pdf/1506.01186.pdf). 

### Training phase

The training history is displayed below for the 100 training epochs, using all the instrument pairs with keyboard_acoustic as origin.   

# Results
>[Table of contents](#table-of-contents)

The numeric value of the loss can serve as a performance metric during training, but the most important part of this applied work is to observe the results subjectively. This section showcases the results both visually and with audios. 

At the end of every training epoch the same audio file has been used to generate a spectrogram frame and the corresponding audio with the target timbre. 


# Conclusion 
>[Table of contents](#table-of-contents)

The system presented in this work can perform the Timbre Transfer problem and achieve reasonable results. However, it is obvious that this system has some limitations and that the results are still far from being usable in a professional music production environment. In this section, the [Results](#results) presented above are discussed.


# Acknowledgements
>[Table of contents](#table-of-contents)  

I'd like to thank

# References
>[Table of contents](#table-of-contents)
