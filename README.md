# cow_identification
python code to apply yolo net on a video sequence to detect the presence of a cow, detect the identification ear-tag and read the number on it.

The script use a yolov3 net trained with custom dataset to detect the ear-tag.

When a tag is detect the easyocr algorithm is applied on the ROI (Region Of Interest)
