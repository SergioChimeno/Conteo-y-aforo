import cv2 as cv

green=(0,255,0)
video_source = '../data/vtest.mov'
video = cv.VideoCapture(video_source)

people_cascade=cv.CascadeClassifier('cascade.xml')

def detect_people_and_count(frame):
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame_h,frame_w= frame.shape[:2]

    people = people_cascade.detectMultiScale(gray_frame,1.5,150)
    for x,y,w,h in people:
        cv.rectangle(frame,(x,y),(x+w,y+h),green,3)
        
    cv.putText(frame,'Aforo: '+str(len(people)),(10,frame_h-30),cv.FONT_HERSHEY_SIMPLEX,1,green,3)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = frame[700:,700:]
    detect_people_and_count(frame)

    cv.imshow('Video', frame)
    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()
video.release()
