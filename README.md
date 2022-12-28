# Music Separated Speech Enhancement using Image Translation 
 
# Table of contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Method](#method)
* [Training](#training)
* [Results](#results)
* [Conclusion](#conclusion)
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

# Method
>[Table of contents](#table-of-contents)

The Pix2Pix architecture has been designed for image processing tasks, but in this case the format of the data is audio. Therefore, a preprocessing step to convert a 1D signal (audio) into a 2D signal (image) is required.


# Dataset
>[Table of contents](#table-of-contents)

Given the description of the problem, the dataset must contain the same audios played by different instruments. Unfortunately, this is very complex to achieve with human performances because of time alignment, note intensity differences, or even instrument tuning changes due to their physical construction. 

For this reason, the audios of the dataset have been synthesized from MIDI files to obtain coherent and reliable data from different instruments. By doing this we ensure that the only change between two audios is the timbre, although this has its own limitations. 

### Dataset download

The dataset has been created using a combination of two publicly available datasets:

* Classical Music MIDI, from Kaggle: https://www.kaggle.com/soumikrakshit/classical-music-midi

* The NSynth Dataset, “A large-scale and high-quality dataset of annotated musical notes”, Magenta Project (Google AI): https://magenta.tensorflow.org/datasets/nsynth

### Alternative dataset

The MAESTRO Dataset contains more than 200 hours of music in MIDI format and can be used to generate an even larger collection of synthesized music. Although the resulting size of the synthesized dataset made it impractical for the scope of this project, the author encourages other researchers with more computing resources to try this option as well. 

* The MAESTRO Dataset “MIDI and Audio Edited for Synchronous TRacks and Organization”, Magenta Project (Google AI): https://magenta.tensorflow.org/datasets/maestro

### Audio synthesis

The audios are generated from these 2 datasets by loading the notes from the MIDI file as a sequence of (pitch, velocity, start_time, end_time). Then, the corresponding note from the NSynth dataset is loaded, modified to the note duration, and placed into an audio file. After repeating these two steps for all the notes in the sequence, the piece from the MIDI file is synthesized as illustrated in this diagram:
<p align="center">
<img src="docs/NoteSynthesizer_diagram.png" width="650" height="450">
</p>
<p align="center">
Audio synthesizer block diagram. The notes from the MIDI file and the notes from NSynth are combined into a synthesized output audio. 
</p>

The procedure has been done with all the MIDI files in [Classical Music MIDI](https://www.kaggle.com/soumikrakshit/classical-music-midi) and with the following instruments from [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) in the note quality 0 (Bright):
* keyboard_acoustic
* guitar_acoustic
* string_acoustic
* synth_lead_synthetic

### Pre/Post processing

The Magnitude Spectrograms are converted from linear domain to logarithmic domain using the function ``amplitude_to_db()`` within the ``data.py`` module, inspired from librosa but adapted to avoid zero-valued regions. The implication of this is that the magnitudes are in decibels (dB), and the distribution of the magnitude values is more similar to how humans hear.  

The minimum magnitude considered to be greater than zero is amin, expressed as the minimum increment of a 16 bit representation (-96 dB).    
```python
amin = 1 / (2**16)
mag_db = 20 * np.log1p(mag / amin)
mag_db /= 20 * np.log1p(1 / amin) # Normalization
```

Finally, the range is normalized in [-1,1] instead of [0,1] using the following conversion:
```python
mag_db = mag_db * 2 - 1
```

To recover the audio, the inverse operations must be performed. Denormalize to [0,1], convert from logarithmic to linear using the function ``db_to_amplitude()`` from ``data.py``, and then compute the inverse STFT using ``librosa.istft()`` with the magnitude and the phase estimations. The complex spectrogram and the final audio can be obtained from the magnitude and phase as: 
```python
S = mag * np.exp(1j * phase)
audio = librosa.istft(S,...)
```

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
