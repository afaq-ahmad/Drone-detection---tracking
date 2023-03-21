1. Changing yolov3.cfg file from cfg folder:

	> change line max_batches to (classes*2000 but not less than 4000), f.e. max_batches=6000 if you train for 3 classes
	> change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
	> set network size width=416 height=416 or any value multiple of 32
	> change line classes=80 to your number of objects in each of 3 [yolo]-layers
	> change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that 
		it only has to be the last [convolutional] before each of the [yolo] layers.So if classes=1 then should be filters=18. 
		If classes=2 then write filters=21.
	
2. Create file obj.names:
	> This file should contain the names of classes.


3. Create file obj.data

	> classes= 1
	> train  = drone_detection/mavicpro2_drone_dataset/train.txt #files contains training images path.
	> valid  = drone_detection/mavicpro2_drone_dataset/test.txt
	> names = drone_detection/mavicpro2_drone_dataset/obj.names
	> backup = drone_detection/mavicpro2_drone_dataset/backup/


4. Yolo coordinate format:
	>Its normalize coordinates divided by image size width and height.

		<object-class> <x_center> <y_center> <width> <height>

		Where:

		<object-class> - integer object number from 0 to (classes-1)
		<x_center> <y_center> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
		for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
		atention: <x_center> <y_center> - are center of rectangle (are not top-left corner)

5. Pretrained Coco Weights:
	Download darknet53.conv.74 pretrained weigths from darknet repo 


6. Calculating anchors:
	> ./darknet detector calc_anchors drone_detection/mavicpro2_drone_dataset/obj.data -num_of_clusters 9 -width 416 -height 416
	> Write the anchors in yolov3.cfg


7. Start Training:
	./darknet detector train drone_detection/mavicpro2_drone_dataset/obj.data drone_detection/mavicpro2_drone_dataset/yolov3.cfg darknet53.conv.74 -map

7. Testing:
	./darknet detector test drone_detection/mavicpro2_drone_dataset/obj.data drone_detection/mavicpro2_drone_dataset/yolov3.cfg drone_detection/mavicpro2_drone_dataset/backup/lastweight
