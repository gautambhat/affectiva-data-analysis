# affectiva-data-analysis
Data preprocessing and neural network-based analysis of Affectiva's video data, for use in evaluating student learning.

----

*Credits to [Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book/tree/master/code/ch12) for providing the implementation of the neural network in his book, Python Machine Learning.*

-----

Our aim was to get all the data that Affectiva's Affdex C++ API provides, and evaluate its performance in detecting student frustration. We also, wanted to validate the accuracy of the emotions it classifies, based on manual marking on videos based on three emotions - frustrated, engaged, and tired.

We processed the data appropriately to be fed into a neural network that we trained to accurately detect when a student is frustrated, engaged, and/or tired. As of now, we are a bit short on data.

**Brief report of our work and results can be found [here](http://cs-people.bu.edu/gautam/cs585/p3/).**

----

To run the project, change lines 22-25 (or however many files there are; I had four. I know, I could have automated the parsing of filenames) of **affdex_data_processing.py** to the path of the csv files where raw video output data is stored.

```df_gb1 = pandas.read_csv('/Path/to/labelled_0.csv')
 23 df_gb4 = pandas.read_csv('/Path/to/labelled_1.csv')
 24 df_gb6 = pandas.read_csv('/Path/to/labelled_2.csv')
 .
 .
 .
 25 df_gb7 = pandas.read_csv('/Path/to/labelled_n.csv')
 ```

and change the the path of the output file according to your filesystem in **line 265** of **affdex_data_processing.py** and **line 349** of **affdex_nn.py**.

----

Contact me for any calrifications or issues. I'll try to reply to the best of my capabilities. `gautam` at `bu` dot `edu`
