import cv2
id=input("enter your id no.")
name=input("enter name of person no.")
cap=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image_no=0
while True:
    ret, frame = cap.read()
    face = face_cascade.detectMultiScale(frame, 1.3, 5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    for (x,y,z,h) in face:
        image_no=image_no+1
        cv2.imwrite("dataset/"+str(name)+"_"+str(id)+"_"+str(image_no)+".jpg",gray[y:y+h,x:x+z])
        cv2.rectangle(frame, (x, y), (x +h, y + z), (0, 0, 255), 15)
        cv2.waitKey(100)
        cv2.imshow("image",frame)
    if(image_no>50):
        break
cap.release()
cv2.destroyAllWindows()

