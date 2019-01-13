##
###snapshot taking


def snapshot(str,rval,frame):
    import cv2    
    cv2.imwrite(str,frame)
    
    return
