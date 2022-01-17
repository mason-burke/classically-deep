# Machine Composed Music using Deep Neural Networks

## notes

This was my final project for my deep learning class, taken in Fall of 2021. Our goal was to take classical music, train a deep net to learn the structure of it, and then have the net be able to create its own music. This project was a failure, as we were not able to make the model learn for some reason. The real triumph was in the data processing. The MusicNet data is a .npz archive, which has a song ID paired with a (IntervalTree, Audio Samples) tuple. The IntervalTree contains information about all of the notes playing at a certain point in time. We decided that the best way to process this was to take samples of a song's IntervalTree at various timesteps, and then mash them up into a NumPy array, which could be fed into the network. This worked relatively well (as best we could check), but something went wrong in our implementation of our RNN. Check out our [DevPost](https://devpost.com/software/machine-composed-music-using-deep-neural-networks), where we submitted this to our class' "Deep Learning Day," a hackathon showcasing all of the deep learning class' final projects.


## structure

We have our neural network itself in the class model.py. Our preprocessing/network training is in
classically_deep.py. This means that all of the model-related parameters are found in the model class,
and the data preprocessing and postprocessing parameters are found in the classically_deep file. We
process our data from MusicNet piece by piece in organize_song(), and then concatenate all tokenized
songs into one numpy array, which can then be saved to a file to facilitate future use. Then, the model
is trained by this input data, and then the trained model is used to generate a song using generate_song().
This can then be used to create a MIDI file using generate_MIDI().

## how to use

To preprocess song data and save it to a file, set save_mode = True in main() of classically_deep.py,
line 212. If you want to load this data and run the model, set save_mode = False. Do note that in order
to run this project, you will need the .npz version of MusicNet, which occupies roughly 11.72 GB of storage.
This dataset can be found [here](https://www.kaggle.com/imsparsh/musicnet-dataset?select=musicnet.npz).
This project has a number of parameters relating to number of training songs, number of samples to take from
each song, how frequently to take samples, etc. These are global variables and are found at the top of
classically_deep.py. This project expects the MusicNet.npz file to be located in a folder called "data/musicnet.npz"
from the directory that classically_deep.py is located in, which is also where the output song will be written.
