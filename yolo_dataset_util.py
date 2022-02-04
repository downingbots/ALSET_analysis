"""
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
 
70/30 train/validation ration

YOLO format, with one *.txt file per image (if no objects in image, no *.txt file is required). The *.txt file specifications are:

    One row per object
    Each row is class x_center y_center width height format.
    Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
    Class numbers are zero-indexed (start from 0)
 
# python3 yolov5-6.0/train.py --img 640 --batch 16 --epochs 5 --data alset_yolov5.yaml --weights yolov5-6.0/yolov5n.pt
"""
import yaml
import os
import cv2
from shutil import copyfile
from utilborders import *
from dataset_utils import *

class yolo_dataset_util():

  def __init__(self):
      self.dsu = DatasetUtils(app_name=None, app_type=None)
      # create directories if not already created
      yolo_dirs = []
      yolo_dirs.append( self.dsu.yolo_dataset_dir() )
      yolo_dirs.append( self.dsu.yolo_images_train_path() )
      yolo_dirs.append( self.dsu.yolo_images_val_path() )
      yolo_dirs.append( self.dsu.yolo_labels_train_path() )
      yolo_dirs.append( self.dsu.yolo_labels_val_path() )
      self.dsu.mkdirs(yolo_dirs)
      yaml_file = os.path.basename(self.dsu.yolo_dataset_yaml())
      try:
        copyfile(yaml_file, self.dsu.yolo_dataset_yaml())
      except Exception as e:
        print("exception:",e)
        pass

      yfile = self.dsu.yolo_dataset_yaml()
      with open(yfile, "r") as stream:
        try:
          self.yolocfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
          print(exc)
      self.class_cnt = []
      for i in range(self.yolocfg["nc"]):
        self.class_cnt.append(0)
      self.train_val_ratio = 70 / 30
      self.yolo_img_sz = 640
      # figure out the current train/val ratio
      self.num_train_images = len(os.listdir(self.dsu.yolo_images_train_path()))
      self.num_val_images = len(os.listdir(self.dsu.yolo_images_val_path()))

  #
  # YOLO format, with one *.txt file per image (if no objects in image, 
  # no *.txt file is required). The *.txt file specifications are:
  #   One row per object
  #   Each row is class x_center y_center width height format.
  #   Box coordinates must be in normalized xywh format (from 0 - 1). 
  # If your boxes are in pixels, divide x_center and width by image width, 
  # and y_center and height by image height.
  #
  # Class numbers are zero-indexed (start from 0)
  #
  def convert_to_yolo_bounding_box(self, bb_label, bb):
      # <object-class> <x> <y> <width> <height>
      # print("self.yolocfg",self.yolocfg, bb_label)
      try:
        label_num = self.yolocfg["names"].index(bb_label)
      except:
         return None
      ctr = bounding_box_center(bb)
      yolo_bb = str(label_num)+ " " + str(ctr[0]/self.yolo_img_sz) + " " + str(ctr[1]/self.yolo_img_sz) + " " + str(bb_width(bb)/self.yolo_img_sz)+ " " +str(bb_height(bb)/self.yolo_img_sz)+"\n"
      # print("yolo_bb:", yolo_bb)
      return yolo_bb

  def get_file_path(self, image, suffix="jpg"):
      train_image, val_image = None, None
      train_label, val_label = None, None
      if suffix == "jpg" or "jpg" in suffix:
        train_image = self.dsu.yolo_images_train_path() + image
        val_image = self.dsu.yolo_images_val_path() + image
      if suffix == "txt" or "txt" in suffix:
        label = image[0,-3] + "txt"
        train_image = self.dsu.yolo_labels_train_path() + label
        val_image = self.dsu.yolo_labels_val_path() + label
      if (train_image is not None and os.path.exists(train_image)):
        return train_image
      if (val_image is not None and os.path.exists(val_image)):
        return val_image
      if (train_label is not None and os.path.exists(train_label)):
        return train_label
      if (val_label is not None and os.path.exists(val_label)):
        return val_label
      return None

  # idempotent function call 
  def record_yolo_data(self, image_path, bb_label, bb):
      image = cv2.imread(image_path)
      image_name = os.path.basename(image_path)
      label_name = image_name[:-3] + "txt"
      scale_factor = self.yolo_img_sz / image.shape[0]
      resized_bb = copy.deepcopy(bb)
      for i in range(4):
        for j in range(2):
          resized_bb[i][0][j] = resized_bb[i][0][j] * scale_factor 
      yolo_bb_line = self.convert_to_yolo_bounding_box(bb_label, resized_bb)
      if yolo_bb_line is None:
        print("skipped non-yolo label:", bb_label)
        return # skip
      img_path = self.get_file_path(image_name)
      if img_path is None:
        dim = (self.yolo_img_sz, self.yolo_img_sz)
        resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        if self.num_val_images == 0 or self.train_val_ratio < self.num_train_images / self.num_val_images:
          img_path = self.dsu.yolo_images_val_path() + image_name
          label_path = self.dsu.yolo_labels_val_path() + label_name
          self.num_val_images += 1
        else:
          img_path = self.dsu.yolo_images_train_path() + image_name
          label_path = self.dsu.yolo_labels_train_path() + label_name
          self.num_train_images += 1
        retkey, encoded_image = cv2.imencode(".jpg", resized_img)
        with open(img_path, 'wb') as f:
          f.write(encoded_image)
      else:
        # if already exists, reuse files
        if "train" in img_path:
          label_path = self.dsu.yolo_labels_train_path() + label_name
        else:
          label_path = self.dsu.yolo_labels_val_path() + label_name
      # create or write bb line to label file associated with image
      with open(label_path, 'a+') as file:
        file.write(yolo_bb_line)

