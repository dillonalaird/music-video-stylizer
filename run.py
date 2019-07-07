import os
import sys
import cv2
import math
import numpy as np
import tensorflow as tf
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

sys.path.append("models/research")
sys.path.append("models/research/object_detection")
from object_detection.utils import ops as utils_ops
from moviepy.video.io.VideoFileClip import VideoFileClip
from argparse import ArgumentParser
from utils import label_map_util
from scipy import fft
from PIL import Image
from tqdm import tqdm

# this must be after import utils from object_detection since
# fast-style-transfor/src also has a utils
sys.path.append("fast-style-transfer/src")
import transform

NUM_CLASSES = 90
# Use this if you want to load the label map
# PATH_TO_LABELS = os.path.join("models/research/object_detection/data",
#                               "mscoco_label_map.pbtxt")

parser = ArgumentParser()
parser.add_argument("--style-model", type=str, required=True, dest="style_model",
                    help="The model used for style transfer (.ckpt file), models can be found" +
                    " in this repo https://github.com/lengstrom/fast-style-transfer")
parser.add_argument("--seg-model", type=str, required=True, dest="seg_model",
                    help="The model used for segmentation (.pb file), models can be found" +
                    " in this repo https://github.com/tensorflow/models/tree/master/research/object_detection")
parser.add_argument("--video-in-path", type=str, required=True, dest="video_in_path",
                    help="The path of the input video")
parser.add_argument("--video-out-path", type=str, default="out.mp4",
                    dest="video_out_path", help="The path of the output video")
parser.add_argument("--device", type=str, default="/gpu:0",
                    help="Device to run the model on, for example '/gpu:0'")
args = parser.parse_args()


def merge_classes(masks, target_class, classes):
    """Takes a numpy array containing different masks for different class
    instances and merges the masks for the `target_class`.

    Parameters
    ----------
    masks : numpy.ndarray
        A numpy array that represents different masks for different classes. It
        has shape `C`, `H`, `W` where `C` is the different class instances.
    target_class : int
        The target class you want to merge.
    classes : numpy.ndarray
        An array representing the different classes present in each of the masks
        across the first (`C`) dimension.

    Returns
    -------
    numpy.ndarray
        A single mask where all masks containing `target_class` were merged.
    """

    locs = np.where(classes == target_class)[0]
    masks = masks[locs, :, :]
    masks = np.max(masks, axis=0)
    return masks


def get_outline(mask):
    """Takes in a mask and returns an image of the outline of the mask.

    Parameters
    ----------
    mask : numpy.ndarray
        A numpy array representing a mask of 0's and 1's.

    Returns
    -------
    numpy.ndarray
        An image containing only the outline of the mask.
    """

    edges = cv2.Canny(mask.astype(np.uint8), 0.5, 1.5)
    dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return dilated


def draw_random_triangles(mask, size=10, samples=10000):
    """Draws random triangles around a mask to add a noise-like effect to the
    edge of the mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The mask image to use.
    size : int, optional
        The size of the trianls.
    samples : int, optional
        The number of random triangles to generate. Note this will only add
        triangles that touch the original mask not triangles generated outside
        of the original mask.

    Returns
    -------
    numpy.ndarray
        The original mask with random triangles added to the edges to make it
        look like noise has been added.
    """

    mask_t = mask.copy()
    # faster than using np.random.choice on masked indices
    for _ in range(samples):
        x = np.random.randint(mask.shape[0])
        y = np.random.randint(mask.shape[1])
        r = np.random.uniform(0, size)
        if mask[x, y] == 1:
            x1 = int(np.random.uniform(-size, size))
            y1 = np.random.choice([1, -1])*int(np.sqrt(size**2 - x1**2))
            x1 = x + x1
            y1 = y + y1

            x2 = int(np.random.uniform(-size, size))
            y2 = np.random.choice([1, -1])*int(np.sqrt(size**2 - x2**2))
            x2 = x + x2
            y2 = y + y2

            x3 = int(np.random.uniform(-size, size))
            y3 = np.random.choice([1, -1])*int(np.sqrt(size**2 - x3**2))
            x3 = x + x3
            y3 = y + y3

            p = np.array([[y1, x1], [y2, x2], [y3, x3]])
            mask_t = cv2.fillPoly(mask_t, [p], 1)

    return mask_t


def draw_random_circles(mask, size=10, samples=10000):
    """Draws random circles around a mask to add a noise-like effect to the edge
    of the mask.

    Parameters
    ----------
    mask : numpy.ndaray
    size : int, optional
    samples : int, optional

    Returns
    -------
    numpy.ndarray
        The original mask with random circles added to the edges to make it look
        like noise has been added.
    """

    mask_t = mask.copy()
    # faster than using np.random.choice on masked indices
    for _ in range(samples):
        x = np.random.randint(mask.shape[0])
        y = np.random.randint(mask.shape[1])
        r = np.random.uniform(0, size)
        if mask[x, y] == 1:
            mask_t = cv2.circle(mask_t, (y, x), size, 1, -1)

    return mask_t


def get_base_bumps(video_clip, f_b=100):
    """Takes in a video clip and returns a normalized array representing the
    amount of base in each video frame, where 0 is no base and 1 is a lot of
    base.

    Parameters
    ----------
    video_clip : moviepy.video.io.VideoFileClip.VideoFileClip
        A moviepy VideoFileClip to extract the base arrays from.
    f_b : int, optional
        The upper limit of frequency to use to calculate the amount of base.

    Returns
    -------
    tuple of list
        A tuple of two lists each representing the amount of base from each
        channel.
    """

    fps_v = video_clip.fps
    audio_clip = video_clip.audio
    raw_clip = audio_clip.to_soundarray()
    fps_a = audio_clip.fps
    window_size = int(fps_a/fps_v)
    num_windows = math.ceil(len(raw_clip)/window_size)
    c1 = [l[0] for l in raw_clip]
    c2 = [l[1] for l in raw_clip]

    result1 = []
    result2 = []
    for i in range(num_windows):
        i_start = i * window_size
        i_end = (i + 1) * window_size
        windowed_sig_FFT1 = abs(fft(c1[i_start:i_end]))
        windowed_sig_FFT2 = abs(fft(c2[i_start:i_end]))

        f_b_ind = int(f_b*len(windowed_sig_FFT1)/fps_a)

        result1.append(sum(windowed_sig_FFT1[1:f_b_ind])**2)
        result2.append(sum(windowed_sig_FFT2[1:f_b_ind])**2)

    # normalize
    result1 /= max(result1)
    result2 /= max(result2)
    return (result1, result2)


def stylize_objects(seg_model_path, orig_path_in, style_path_in, path_out,
                    device_t="/gpu:0", target_class=1):
    """Generates a video where objects are segmented out and stylized. An
    outline is also drawn around the person and noise is added in proportion to
    the amount of base.

    Parameters
    ----------
    seg_model_path : str
        The path to the segmentation model. Should be a .pb file.
    orig_path_in : str
        The path to the original un-stylized video file.
    style_path_in : str
        The path to the stylized video file.
    path_out : str
        The path to save the new video with only the objects stylized.
    device_t : str, optional
        The device to run the network on.
    target_class : int, optional
        The target you want generate masks for and stylize.

    Example
    -------
    stylize_objects("models/model.pb", "video.mp4", "inter_styled_video.mp4",
                    "styled_video.mp4")
    """
    video_clip = VideoFileClip(orig_path_in, audio=True)
    style_video_clip = VideoFileClip(style_path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size,
                                                    video_clip.fps,
                                                    codec="libx264",
                                                    preset="medium",
                                                    bitrate="2000k",
                                                    audiofile=orig_path_in,
                                                    threads=None,
                                                    ffmpeg_params=None)
    ch1, ch2 = get_base_bumps(video_clip)

    # load model
    g = tf.Graph()
    with g.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(seg_model_path, "rb") as f:
            serialized_graph = f.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    # code adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    with g.as_default(), g.device(device_t), tf.Session() as sess:
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ["num_detections", "detection_boxes", "detection_scores",
                    "detection_classes", "detection_masks"]:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)

        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
        detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
        # Reframe is required to translate mask from box coordinates to image
        # coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, video_clip.size[1],
            video_clip.size[0])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict["detection_masks"] = tf.expand_dims(
            detection_masks_reframed, 0)

        image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

        pbar = tqdm(total=int(video_clip.fps * video_clip.duration))
        for i, (frame, style_frame), in enumerate(zip(video_clip.iter_frames(),
                                                      style_video_clip.iter_frames())):
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor:
                                       np.expand_dims(frame, 0)})
            # assume batch size = 1
            classes = output_dict["detection_classes"][0][:int(output_dict["num_detections"][0])]
            # if no target class then have to use a 0 mask
            if target_class not in classes:
                mask = np.zeros((video_clip.size[1], video_clip.size[0]))
                to_style_frame = False
            else:
                mask = merge_classes(output_dict["detection_masks"][0, :, :, :], 1,
                                     classes)
                to_style_frame = True
            mask = draw_random_triangles(mask, size=(ch1[i]*30 + 1e-8))

            outline = Image.fromarray(get_outline(mask))
            mask = Image.fromarray(255*mask)
            nframe = Image.fromarray(frame)
            # can't paste with 0 mask
            if to_style_frame:
                nframe.paste(Image.fromarray(style_frame), mask=mask)
                nframe.paste(outline, mask=outline)

            video_writer.write_frame(nframe)
            pbar.update(1)

        pbar.close()
        video_writer.close()


def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    """Creates a stylized video. Code from lengstrom's repo found here:
    https://github.com/lengstrom/fast-style-transfer
    and the specific file is found here:
    https://github.com/lengstrom/fast-style-transfer/blob/master/evaluate.py

    Parameters
    ----------
    path_in : str
        The path to the video to read in to stylize.
    path_out : str
        The path to save the stylized video.
    checkpoint_dir : str
        The checkpoint dir holding the neural style transfer model. This should
        be a .ckpt file.
    device_t : str, optional
        The device you want to run the model on.
    batch_size : int, optional
        The batch size you want to use for the model.
    """

    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out,
                                                    video_clip.size,
                                                    video_clip.fps,
                                                    codec="libx264",
                                                    preset="medium",
                                                    bitrate="2000k",
                                                    audiofile=path_in,
                                                    threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        pbar = tqdm(total=int(video_clip.fps * video_clip.duration))
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                pbar.update(frame_count)
                frame_count = 0


        if frame_count != 0:
            style_and_write(frame_count)
            pbar.update(frame_count)

        pbar.close()
        video_writer.close()


if __name__ == "__main__":
    # This loads the category index incase you want to stylize some other object
    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map,
    #         max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    name, ext = os.path.splitext(args.video_out_path)
    style_out_path = name + "_styled" + ext
    ffwd_video(args.video_in_path, style_out_path, args.style_model, args.device, 1)

    stylize_objects(args.seg_model, args.video_in_path, style_out_path,
                    args.video_out_path, device_t=args.device)
    os.remove(style_out_path)
