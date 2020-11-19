import cv2 as cv

yellow = (0, 255, 255)
video_source = '../data/vtest.avi'
video = cv.VideoCapture(video_source)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


def detect_people_and_count(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_h, frame_w = frame.shape[:2]

    people, weights = hog.detectMultiScale(frame,winStride=(8, 8),scale=1.06)

    for x, y, w, h in people:
        cv.rectangle(frame, (x, y), (x + w, y + h), yellow, 3)

    cv.putText(frame, 'Aforo: ' + str(len(people)), (10, frame_h - 30), cv.FONT_HERSHEY_SIMPLEX, 1, yellow, 3)


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    detect_people_and_count(frame)

    cv.imshow('Video', frame)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
video.release()
