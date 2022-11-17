cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)