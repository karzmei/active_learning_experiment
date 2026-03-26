# Active Learning Experiment
This repo aim to explore some active learning methods.

## The data:
I chose to work with the EuroSAT dataset, which consists of 27K satelite 64x64 images of Europe. The images were manually labeled and contain 10 classes.
See details here: https://arxiv.org/abs/1709.00029.

Why this data? It's visually nice, easily available via pytorch, and could be a possible active learning use-case, as labeling such images can be non-trivial work. It was also manually labeled and carefully checked, so is relatively "clean" - which can be a downside, since it resembles less a real-world data (which is an awkward thing to say about satelite images). 
BUT it does provide a nice and non-noisy playground for active learning experiments.

Here is a sample of the data (one can immediately see that distibguishing between the classes can be hard):
<img width="1489" height="479" alt="data_sample" src="https://github.com/user-attachments/assets/384042f0-3f24-4b4c-b7bb-62dc1f3567b2" />
