# Udacity Intro to Machine Learning

Notes for myself so I can remember what I did

## 01 Gaussaian Naive Bayes Deployment on Terrain Data

Can only be run on udacity. I did not take the time to get all the imports working. Best run on udacity


## 02 Calculating Naive Bayes Accuracy

Commented out imports and removed files not needed to allow example to run in command line vs only on udacity. "app.py" can be run directly from python as long as you have all the libraries installed or run the Docker image

## 03 Author ID Accuracy

I cloned https://github.com/udacity/ud120-projects to a completely separate directory structure. I've copied the tools directory from ud120-project here to get each assignment to work. I've excluded uploading maildir/* as that is a waste of space and time.

To get Docker to work, need to mount tools and maildir into Docker image. See https://docs.docker.com/docker-for-mac/osxfs/

* tools is located here ~/Documents/Udacity/ud120 Intro to Machine Learning/tools
* maildir is located here ~/Documents/Udacity/ud120 Intro to Machine Learning/maildir

And the run file in the directory shows how I ran it. This start with assignment 3


