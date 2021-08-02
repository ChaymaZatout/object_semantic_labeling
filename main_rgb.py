"""
Name : main_rgb.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 17/07/21 04:14 م
Desc:
"""
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import open3d as o3d
from basisr5x5.simulator import BASISR55

if __name__ == '__main__':
    # classes to map from imagenet to our labels:
    classes = {'barber_chair': 0, 'folding_chair': 0, 'rocking_chair': 0,
               'dining_table': 1, 'pool_table': 1, 'desk': 1,
               'wardrobe': 2,
               'sliding_door': 3,
               'window_shade': 4}
    previous = ''
    is_init = True

    # basisr:
    # visualization:
    vis = o3d.visualization.Visualizer()
    vis.create_window("BASISR 5x5", width=640, height=480)
    vis.get_render_option().background_color = [0.75, 0.75, 0.75]
    vis.get_render_option().mesh_show_back_face = True

    # create BASISR and add it to the visualizer:
    basisr = BASISR55(size=0.15,  # the base size : size x size
                      pins_per_line=5,  # the number of pins per line (colomn)
                      pins_R=0.020,  # the pins R (R = r*2)
                      base_height=0.075,  # the base height
                      pins_height=0.002
                      )
    vis.add_geometry(basisr.base)
    # Add pins to visualizer:
    for j in range(basisr.pins_per_line):
        for i in range(basisr.pins_per_line):
            vis.add_geometry(basisr.pins[j][i])
    # display device:
    vis.poll_events()
    vis.update_renderer()

    # define a video capture object
    vid = cv2.VideoCapture(0)
    ret, frame = vid.read()
    h, w = frame.shape[:2]

    # vgg16:
    model = VGG16(weights='imagenet', include_top=True)

    while (True):

        # Capture the video frame by frame
        ret, frame = vid.read()
        # resize:
        y = (w - h) // 2
        frame = frame[:, y:y + h]
        frame = cv2.resize(frame, (224, 224))
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Classification:
        x = preprocess_input(frame)
        x = x.reshape(1, 224, 224, 3)
        y_pred = model.predict(x)
        labels = decode_predictions(y_pred)
        label = labels[0][0]
        print('%s (%.2f%%)' % (label[1], float(label[2]) * 100))

        # display on basisr:
        if label[1] in classes.keys():
            if label[1] != previous:
                basisr.init_pins()
                if(classes[label[1]]==0):
                    basisr.chair(3)
                elif(classes[label[1]]==1):
                    basisr.table(3)
                elif(classes[label[1]]==2):
                    basisr.dresser(3)
                elif(classes[label[1]]==3):
                    basisr.door(3)
                elif(classes[label[1]]==4):
                    basisr.window(3)
                # update pins:
                for p in basisr.pins.flatten():
                    vis.update_geometry(p)

                previous = label[1]
                is_init = False
        else:
            if is_init == False:
                basisr.init_pins()
                is_init = True
                previous = ""
                # display
                for p in basisr.pins.flatten():
                    vis.update_geometry(p)
        # display on open3d:
        vis.poll_events()
        vis.update_renderer()

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
