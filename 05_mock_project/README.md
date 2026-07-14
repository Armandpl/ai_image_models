# Inspiration for Project

At the end of this class, you will present your best model, the images it can generate and the experiments you ran to get there.
The model doesn't have to be good, the point is to be able to attribute positive changes, by rigorously tracking experiments.
So far, you should have choosen a dataset, trained a first model and ran a sweep over the learning rate.

Here's examples of things you can change, feel free to do your own research to find other ideas, and feel free to plug them in your favourite LLM to expand:
- the model architecture (CNN, tranformers...) - easy
- any hyper-parameters - easy
- the sampling schedule - hard
- the loss - hard
- [...]

Here we will run a simple experiment. We will swap the model architecture from a MLP to a CNN, and investigate if it improves the generations.
