# -*- coding: utf-8 -*-
"""
Yksinkertainen shape detection OpenCV:tä käyttäen, tunnistaa
3-8 kulmaiset objektit ja muut luokitellaan ympyröiksi. 
Objektin viimeisen kulman täytyy yhdistyä ensimmäiseen eli kuvion "alkukohtaan".

@author: Tuomas Ylönen & Anni Halminen - PTIVIS19H
"""

import cv2


img = cv2.imread('test4.jpg')

# Turns the used image into graycoloured
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Uses threshold binary type for threshold operation
_, threshold = cv2.threshold(gray, 136, 255, cv2.THRESH_BINARY)

# Detects corners of image objects
contours, _  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

for contour in contours:
    if i == 0:
        i = 1
        continue

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        cx = int(M['m10']/M['m00']*0.95)
        cy = int(M['m01']/M['m00'])

    # putting shape name at center of each shape
    # Shapes defined by the number of corners and puts recognized identity inside the object
    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar >= 0.95 and ar <= 1.05:
            cv2.putText(img, 'Square', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)
        else:
            cv2.putText(img, 'Quadrilateral', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)

    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)

    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)

    elif len(approx) == 7:
        cv2.putText(img, 'Heptagon', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)
    
    elif len(approx) == 8:
        cv2.putText(img, 'Octagon', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96), 2)
   
    else:
        cv2.putText(img, 'circle', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (96, 96, 96), 2)

# displaying the image after drawing contours
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()