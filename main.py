from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2


# trace the direction of the movement of cat
def tracing(enterY,leaveY):
    if enterY <= 150 and leaveY >= 150: # enter
        return 1
    elif enterY >= 150 and leaveY <= 150:   # leave
        return -1
    else:
        return 0


# constrcut the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min_area", type=int, default=5000, help="minimum area size")
args=vars(ap.parse_args())


if args.get("video", None) is None:     # if the video argument is None, then we are reading from camera
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:                                   # otherwise, we are reading from a video file
    vs = cv2.VideoCapture(args["video"])

firstFrame = None   # all subsequent frames are compared with the first frame and search for significant changes

occupy = False  # true if there is significant movement on the frame, false otherwise

totalCat = 0    # number of cats beneath the car

enterY = currentY = -1

print("looping start")
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    frame = vs.read()
    frame = frame if args.get("video",None) is None else frame[1]
    text = "Unoccupied"
    if frame is None: break # if the frame could not be grabbed, then we have reached the end of the video

    # resize the frame, convert it into grayscale and blur it
    frame = imutils.resize(frame, height=300,width=500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # convert frame image into grey scale image
    gray = cv2.GaussianBlur(gray,(21,21),0) # apply Gaussian smoothing to average pixel intensities across an 21*21 resion,
                                            # this helps smooth out high frequency noice that could throw our motion detection algorithm off.

    # initialize the first frame
    if firstFrame is None:
        firstFrame = gray
        continue

    # compare with first frame and identify motion
    frameDelta = cv2.absdiff(firstFrame, gray)      # compute the absolute difference between the current frame and first frame
    thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh,None,iterations=2)   # dilate the threshold image to fill in holes, then find contours on threshold image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    maxContour = None

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]: continue  # if the contour is too small, ignore it
        if maxContour is None or cv2.contourArea(c) > cv2.contourArea(maxContour):  # filter out the contour with maximum area
            maxContour = c
        (x,y,w,h) = cv2.boundingRect(c)                     # compute the bounding box for the contour
        # print(x," ",y)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)    # draw the contour on the frame
        text = "occupied"  # update text

    if maxContour is None:
        if occupy:
            print("check")
            totalCat += tracing(enterY, currentY)
            print("enter at y = ", enterY)
            print("leave at y = ", currentY)
            occupy = False
    else:
        if not occupy:
            enterY = y
            occupy = True
        currentY = y

    print()
    cv2.putText(frame,"Surronding: {}".format(text),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.putText(frame, "Total cat under car: {}".format(str(totalCat)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh",thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key==ord("q"): break

# cleanup the camera and close any open windows
vs.stop() if args.get("video",None) is None else vs.release()
cv2.destroyAllWindows()