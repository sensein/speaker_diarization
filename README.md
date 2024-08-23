## Experiments in face analysis

This repository contains some preliminary experiments focused on face analysis from video, covering the following key objectives:

- **Person tracking & identification:** Track a person in a video and potentially identify them. This codebase brings us close to achieving that goal, which I know would make Yibei very happy. 
One suggested improvement (I didn't have the time to implement it, sorry): currently, we cluster faces based solely on their embeddings, without considering their location within the frame. To enhance accuracy, we could incorporate the face’s position in the frame. A proposed approach is to use a sliding window over time combined with a majority voting algorithm to assign labels.

- **Active speaker detection:** Once faces are tracked in the video, you can apply speaker diarization (commonly referred to as active speaker detection in the video space). The model available [here](https://github.com/Junhua-Liao/Light-ASD/tree/main) performs quite well for this task and combines audio+video.

- **Facial descriptor extraction:** Extract a vector of descriptors from each face, including sex, age, emotion, speaking flag (yes/no), and Facial Action Coding System (FACS) units. When combined with the speech vector that we have already discussed within senselab, this may enable some cool multimodal analyses.

Once we will have a robuster expertise with these models, we should move our functions and pipelines to senselab (@jordan, feel free to start).

### Getting started with this example

1. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

2. To run the experiments:
   ```bash
   cd src
   python main.py
   ```
