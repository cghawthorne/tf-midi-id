# tf-midi-id
Proof of concept TensorFlow Neural Net to classify midi files.
Based on the [fully connected MNIST example](https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) from TensorFlow.

Reads MIDI files in [midicsv](http://www.fourmilab.ch/webtools/midicsv/) format.

The conversion process from MIDI to an array suitable for machine learning is:
* Split the MIDI event sequence into "windows" of time, 100ms by default.
* Create a 100-element array for every window (should be enough note values to cover most keyboard music).
* For every "note on" event that occurs within a window, increment the position in its array that corresponds to the note value.
* Group the windows into samples, 10 seconds by default.

The system also artifically increases the amount of data available by repeatedly shifting the data by 1ms.

Here's a test run that classifies some Bach preludes and fugues from the The Well-Tempered Clavier against some childrens' songs. I got the midi files from http://www.mfiles.co.uk/ and then ran them through [midicsv](http://www.fourmilab.ch/webtools/midicsv/).

After ~1300 steps, it achieves 100% accuracy with the training, validation, and test sets!

```
Extracting ../musicid/midis/train
Read 31400 windows from ../musicid/midis/train/bach/book1-fugue02.mid.csv
Read 76900 windows from ../musicid/midis/train/bach/book1-fugue24.mid.csv
Read 35000 windows from ../musicid/midis/train/bach/book1-prelude01.mid.csv
Read 38800 windows from ../musicid/midis/train/bach/book1-prelude02.mid.csv
Read 39700 windows from ../musicid/midis/train/bach/book1-prelude03.mid.csv
Read 26300 windows from ../musicid/midis/train/bach/book1-prelude06.mid.csv
Read 96100 windows from ../musicid/midis/train/bach/book1-prelude24.mid.csv
Read 27800 windows from ../musicid/midis/train/bach/book2-fugue02.mid.csv
Read 57200 windows from ../musicid/midis/train/bach/book2-prelude02.mid.csv
Read 72000 windows from ../musicid/midis/train/bach/book2-prelude12.mid.csv
Read 5900 windows from ../musicid/midis/train/children/bye-baby-bunting.mid.csv
Read 12100 windows from ../musicid/midis/train/children/here-we-go-round-the-mulberry-bush.mid.csv
Read 5500 windows from ../musicid/midis/train/children/hey-diddle-diddle.mid.csv
Read 5500 windows from ../musicid/midis/train/children/hickory-dickory-dock.mid.csv
Read 7800 windows from ../musicid/midis/train/children/hot-cross-buns.mid.csv
Read 5900 windows from ../musicid/midis/train/children/lavenders-blue.mid.csv
Read 3700 windows from ../musicid/midis/train/children/mary-mary-quite-contrary.mid.csv
Read 11700 windows from ../musicid/midis/train/children/ring-a-ring-o-roses.mid.csv
Read 12100 windows from ../musicid/midis/train/children/see-saw-margery-daw.mid.csv
Read 24000 windows from ../musicid/midis/train/children/sing-a-song-of-sixpence.mid.csv
Read 11900 windows from ../musicid/midis/train/children/twinkle-twinkle-little-star.mid.csv
Extracting ../musicid/midis/validation
Read 60100 windows from ../musicid/midis/validation/bach/book1-fugue14.mid.csv
Read 24200 windows from ../musicid/midis/validation/bach/book1-prelude14.mid.csv
Read 7800 windows from ../musicid/midis/validation/children/goosey-goosey-gander.mid.csv
Read 12100 windows from ../musicid/midis/validation/children/rock-a-bye-baby-tune-a.mid.csv
Read 12100 windows from ../musicid/midis/validation/children/rock-a-bye-baby-tune-b.mid.csv
Read 11700 windows from ../musicid/midis/validation/children/rub-a-dub-dub.mid.csv
Extracting ../musicid/midis/test
Read 70800 windows from ../musicid/midis/test/bach/book2-fugue07.mid.csv
Read 81200 windows from ../musicid/midis/test/bach/book2-prelude07.mid.csv
Read 5900 windows from ../musicid/midis/test/children/humpty-dumpty.mid.csv
Read 5500 windows from ../musicid/midis/test/children/little-jack-horner.mid.csv
Read 5500 windows from ../musicid/midis/test/children/little-miss-muffet.mid.csv
Read 7800 windows from ../musicid/midis/test/children/the-muffin-man.mid.csv
Step 0: loss = 0.70 (0.228 sec)
Step 100: loss = 0.30 (0.214 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 4958  Precision @ 1: 0.8263
Validation Data Eval:
  Num examples: 1200  Num correct: 843  Precision @ 1: 0.7025
Test Data Eval:
  Num examples: 1700  Num correct: 1520  Precision @ 1: 0.8941
Step 200: loss = 0.23 (0.256 sec)
Step 300: loss = 0.19 (0.527 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 5523  Precision @ 1: 0.9205
Validation Data Eval:
  Num examples: 1200  Num correct: 804  Precision @ 1: 0.6700
Test Data Eval:
  Num examples: 1700  Num correct: 1483  Precision @ 1: 0.8724
Step 400: loss = 0.14 (0.262 sec)
Step 500: loss = 0.16 (0.217 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 5917  Precision @ 1: 0.9862
Validation Data Eval:
  Num examples: 1200  Num correct: 1176  Precision @ 1: 0.9800
Test Data Eval:
  Num examples: 1700  Num correct: 1641  Precision @ 1: 0.9653
Step 600: loss = 0.10 (0.615 sec)
Step 700: loss = 0.10 (0.216 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 5935  Precision @ 1: 0.9892
Validation Data Eval:
  Num examples: 1200  Num correct: 1200  Precision @ 1: 1.0000
Test Data Eval:
  Num examples: 1700  Num correct: 1691  Precision @ 1: 0.9947
Step 800: loss = 0.05 (0.277 sec)
Step 900: loss = 0.06 (0.664 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 5980  Precision @ 1: 0.9967
Validation Data Eval:
  Num examples: 1200  Num correct: 1200  Precision @ 1: 1.0000
Test Data Eval:
  Num examples: 1700  Num correct: 1699  Precision @ 1: 0.9994
Step 1000: loss = 0.05 (0.334 sec)
Step 1100: loss = 0.03 (0.232 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 5998  Precision @ 1: 0.9997
Validation Data Eval:
  Num examples: 1200  Num correct: 1200  Precision @ 1: 1.0000
Test Data Eval:
  Num examples: 1700  Num correct: 1700  Precision @ 1: 1.0000
Step 1200: loss = 0.04 (0.620 sec)
Step 1300: loss = 0.04 (0.234 sec)
Training Data Eval:
  Num examples: 6000  Num correct: 6000  Precision @ 1: 1.0000
Validation Data Eval:
  Num examples: 1200  Num correct: 1200  Precision @ 1: 1.0000
Test Data Eval:
  Num examples: 1700  Num correct: 1700  Precision @ 1: 1.0000
```
