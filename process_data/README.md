# Processing data

We can either synthesize data or process existing data into our pipeline. Either way, for each corpus of music data, we create three directories: ```train```, ```valid```, and ```test```. In each directory, we need a ```meta.p``` file. This file is created by running either ```synthesis.sh``` or ```measure_segmenter.sh```. In fact, ```meta.p``` file stores all event data; the MIDI files outputted are for ease of inspection. 

You can create training examples using a different method than the two here, so long as you create a ```meta.p``` file. Currently, all that is required for ```main.py``` are the ```origs```, ```measure_sdm```, ```measure_boundaries```, ```ts```, and ```f``` (filename) fields.


## Synthesizing data

Run ```synthesis.sh```. This takes all ```.mid``` files in a given directory (already split into train, valid, test), extracts all measures, and then creates new melodies according to a given structure. A melody will be synthesized using existing measures, all within the same time signature. For example, we can create melodies with an [A,B,A] and [A,B,B] structure. See the argparse flags in ```synthesis.py```.

Truly synthetic music generation (each note is random, or decided completely by an algorithm) was not implemented.

## Preprocessing existing data

First, we transpose everything to C Major/A minor, which effectively increases the amount of data we have. Run ```transpose.sh``` with the necessary command-line flags.

Then, we want to segment existing songs into measures. Run ```measure_segmenter.sh``` with the necessary flags. Unfortunately, you may have to finick a bit with this, since each corpus has some quirks. For instance, the Nottingham corpus I used has a pickup into the first measure that you have to account for in parsing for measures. Note that we skip songs with multiple time signatures, which can be easily changed in the future.

Note that all of this code is designed for monophonic melodies!

## Processed data

In the [music_data](../music_data/) folder, there is some already-preprocessed data.

The [CMaj_Nottingham](../music_data/CMaj_Nottingham) directory has been transposed and measure-segmented already.

I'm working on transposing the [guitar](../music_data/guitar) directory, which is the "Classical_Guitar_classicalguitarmidi.com_MIDIRip" section of the [Reddit MIDI collection](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/).
