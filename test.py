# 下面是python-OpenCv调用IP摄像头的方法
# 2021年12月15日21:22:27
import cv2

if __name__ == "__main__":
    # IP视频地址
    url1 = "http://admin:admin@192.168.2.172:8081"
    url2 = "http://admin:admin@192.168.2.121:8081"
    cap1 = cv2.VideoCapture(url1)
    cap2 = cv2.VideoCapture(url2)
    while cap1.isOpened() and cap2.isOpened():
        # Capture frame-by-frame
        ret1, frame1 = cap1.read()
        # Display the resulting frame
        cv2.imshow('frame1', frame1)
                # Capture frame-by-frame
        ret2, frame2 = cap2.read()
        # Display the resulting frame
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # When everything done, release the capture
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

