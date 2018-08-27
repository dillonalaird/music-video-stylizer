# Music Video Styler
The music video styler program uses neural style transfer and image segmentation
to only add the neural style transfer to people or objects in the video. A line
is also drawn around the objects and noise is added in proportion to the amount
of low frequency content or base.


## Installation Instructions
The music video styler uses two main repos: tensorflow's [object detection
repository](https://github.com/tensorflow/models/tree/master/research/object_detection)
and lengstrom's [fast style transfer repository](https://github.com/lengstrom/fast-style-transfer).
Please be sure to follow the object detection [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
after you clone the repository. In order to run the program you must first
download models that work with both repositories.

* tensorflow's models can be found from their
  [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
  Make sure to download a model that can output masks.
* lengstrom's models can be found
  [here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ).


## Example
`ipython run.py -- --style-model style-transfer-models/udnie.ckpt --seg-model mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --video-in-path video/music_video.mp4 --video-out-path styled_music_video.mp4`
