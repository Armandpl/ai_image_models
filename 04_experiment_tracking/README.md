# 04 Experiment Tracking

To improve models, we need to run experiments. It is important to record:
- which code we used to train
- what where the parameters/hyper-paramters
- what data the model was trained on
as well as:
- how the training went: loss over time, system metrics (e.g gpu RAM)
- test results, in our case FID

If we dont track those things, it's super easy to get confused and not attribute results to the correct causes.
Also super easy to write bugs, and not realize.
