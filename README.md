# rtt_tracking

This repo provides a solution for the Rotating Table Task (RTT) in the Robocup@work competition

### Setup your environment


### Usage

To evaluate the results of your tracker, you can run the `eval_metrics.py` file which reports the performance of the tracker. You have to provide it the name of the output `.json` file.
The groundtruth annotations are stored in the `data/` folder by default.The structure of the test folder is as follows

    data
    ├── images                  # Contains the .jpg images
    └── labels                  # Contains labels, each frame has a .json file

```python eval_metrics.py --tracker_outputs name_of_tracker_output_file.json```
