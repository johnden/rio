#!/usr/bin/python
#encode:utf8
#from __future__ import division
import os
import argparse
import cv2
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import LSWMS

pi = math.pi
EnableOfDisplay = "on" # on for show image

'''
@line_size
'''
def line_size(line):
    return math.sqrt((line[0][0]-line[1][0])**2 \
    + (line[0][1]-line[1][1])**2)   

'''
@zpoint
'''
def zpoint(pij, v, d):
    out_point = [0,0]
    out_point[0] = pij[0] + v[0]*d
    out_point[1] = pij[1] + v[1]*d
    return out_point

'''
@prpoint
'''
def prpoint(center, di, dj, vi, vj):
    qi = [0,0]
    qj = [0,0]
    qi[0] = center[0] + (di*(vi[0]/math.fabs(math.sqrt(vi[0]**2+vi[1]**2))))
    qi[1] = center[1] + (di*(vi[1]/math.fabs(math.sqrt(vi[0]**2+vi[1]**2))))
    qj[0] = center[0] + (dj*(vj[0]/math.fabs(math.sqrt(vj[0]**2+vj[1]**2))))
    qj[1] = center[1] + (dj*(vj[1]/math.fabs(math.sqrt(vj[0]**2+vj[1]**2))))
    pij   = [0,0]
    pij[0] = qi[0] + (dj*(vj[0]/math.fabs(math.sqrt(vj[0]**2+vj[1]**2))))
    pij[1] = qi[1] + (dj*(vj[1]/math.fabs(math.sqrt(vj[0]**2+vj[1]**2))))

'''
@power
'''
def power(src, angle, a, b):
    dx = (b[0]-a[0])/math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    dy = (b[1]-a[1])/math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    sum = 0.0
    n   = 0
    out = copy.copy(src)
    
    for i in range(int(math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2))):
        y = a[1] + i*dy
        x = a[0] + i*dx
        
        if angle[y][x]*(180.0/pi)<361 and angle[y][x]*(180.0/pi)>-1:
            e = math.acos(abs(dx*math.cos(angle[y][x]) + dy*math.sin(angle[y][x])))*(180.0/pi)
            sum = sum + e
            n = n + 1
            ddx = 30*math.cos(angle[y][x])
            ddy = 30*math.sin(angle[y][x])
            
            cv2.circle(out, (x,y),2,(0,255,255),2,8)
            cv2.line(out, (x,y),(x+ddx,y+ddy),(0,255,255),2,8)
            cv2.line(out, a,b,(255,255,0),2,8)
    if n==0:
        return 90
    else:
        return sum/float(n)

'''
@main function
'''
# def  main():
if __name__ == "__main__":
    # main function    
    image1               = cv2.imread('002.jpg')
    height,width,channel = image1.shape
    image                = cv2.resize(image1, ((int)(width/8), (int)(height/8)), interpolation=cv2.INTER_NEAREST)
    
    if EnableOfDisplay == "on":
        cv2.imshow("SrcImg", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    out = np.zeros(np.shape(image),np.typeNA['B'])
    out = copy.copy(image)
    
    blur_out = copy.copy(out)
    out_x = copy.copy(out);out_y = copy.copy(out)
    blur_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_out = cv2.blur(blur_out,(5,5))
    
    Tg = 20.0; BetaTg = 30.0; LumTg  = 40.0 # Tg BetaTg LumTg a,b,c
    R  = 1
    numMaxLSegs = 10000
    # print ("blur_out height %d, width %d.\n" % (np.shape(blur_out)[0], np.shape(blur_out)[1]))
    if EnableOfDisplay == "on":
        cv2.imshow("BlurOut", blur_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    errors   = []    # [0.0]
    lSegs    = []    # [[(0,0)]]
    outlines = []    # [[(0.0)]]

    lswms                = LSWMS.LSWMS(np.shape(blur_out)[0:2], R, numMaxLSegs, False)
    retLS, lSegs, errors = lswms.run(image, lSegs, errors)                              # core

    # lSegs
    # Show lSegs for image
    for i in range(0,len(lSegs)):
        cv2.line(image, lSegs[i][0], lSegs[i][1], (0,0,255), 2)
        cv2.imshow("line",image)
        cv2.waitKey(0)

    floody = copy.copy(blur_out)
    floodx = copy.copy(blur_out)
    
    floody.fill(0)
    floodx.fill(0)
    
    for i in range(len(lSegs)):
        if abs(lSegs[i][0][1]-lSegs[i][1][1])>abs(lSegs[i][0][0]-lSegs[i][1][0]) and line_size(lSegs[i])>100:
            cv2.line(floody,lSegs[i][0],lSegs[i][1],(255,255,255),3,8)
        elif line_size(lSegs[i])>100:
            cv2.line(floodx,lSegs[i][0],lSegs[i][1],(255,255,255),3,8)
    
    contours  = [] # (0,0)
    contoursy = [] # (0,0)
    contoursx = [] # (0,0)
    
    contoursy, hierarchyy = cv2.findContours(floody,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contoursx, hierarchyx = cv2.findContours(floodx,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    oolines = [] # [[(0,0)]]
    rows,cols = np.shape(out)[0:2]
    
    for m in range(len(contoursy)):
        ssline = [] # [(),(),(),()]
        ssline = cv2.fitLine(contoursy[m],cv2.cv.CV_DIST_L2,0,0.01,0.01)
        x0 = 0; y0 = 0; x1 = 0; y1 = 0
        if ssline[0] != 0:
            if ((ssline[1]/ssline[0])*ssline[2]-ssline[3])/(ssline[1]/ssline[0])<0:
                x0 = 0
                y0 = ssline[3] - (ssline[1]/ssline[0])*ssline[2]
            else:
                x0 = ((ssline[1]/ssline[0])*ssline[2]-ssline[3])/(ssline[1]/ssline[0])
                y0 = 0
            
            if ((ssline[1]/ssline[0])*ssline[2]+rows-ssline[3])/(ssline[1]/ssline[0])>cols:
                x0 = cols
                y0 = ssline[3] + (ssline[1]/ssline[0])*(cols - ssline[2])
            else:
                x1 = ((ssline[1]/ssline[0])*ssline[2]+rows-ssline[3])/(ssline[1]/ssline[0])
                y1 = rows
        else:
            x0 = ssline[2]
            y0 = 0
            x1 = ssline[2]
            y1 = rows
            
        outline = [] #
        outline.append((x0,y0))
        outline.append((x1,y1))
        oolines.append(outline)
            
    out_x = cv2.Sobel(blur_out,cv2.CV_32F,1,0,3)
    out_y = cv2.Sobel(blur_out,cv2.CV_32F,0,1,3)
    
    magnitude = np.zeros(np.shape(out),np.typeNA['B'])
    angle     = np.zeros(np.shape(out),np.float32)
    
    cv2.cartToPolar(out_x,out_y,magnitude,angle,False)
    
    seed_points = []; vs = []
    
    for i in range(0,len(oolines)):
        v = [0.0,0.0]; vt = [0.0,0.0]; fvt = [0.0,0.0]
        v[0] = (float(oolines[i][0][0])-float(oolines[i][1][0]))/line_size(oolines[i])
        v[1] = (float(oolines[i][0][1])-float(oolines[i][1][1]))/line_size(oolines[i])
        vt[0] = -v[1]
        vt[1] = v[0]
        fvt[0]= -vt[0]
        fvt[1]= -vt[1]
        
        center = [0.0,0.0]
        center[0] = (oolines[i][1][0]+oolines[i][0][0])*0.5
        center[1] = (oolines[i][1][1]+oolines[i][0][1])*0.5
        
        dx = float(oolines[i][1][0]-center[0])/(line_size(oolines[i])*0.5)
        dy = float(oolines[i][1][1]-center[1])/(line_size(oolines[i])*0.5)
    
        for j in range(0,line_size(oolines[i])*0.5,50):
            zp = [0.0,0.0]
            zp[1] = center[1] + j*dy
            zp[0] = center[0] + j*dx
            seed_points.append(zpoint(zp,vt,20))
            seed_points.append(zpoint(zp,fvt,20))
            vs.append(v)
            vs.append(v)
            
    outimage = copy.copy(image)
    
    for i in range(int(len(seed_points))):
        t = float(cv2.getTickCount)
        v = [0.0,0.0]; vt = [0.0,0.0]; fv = [0.0,0.0]; fvt = [0.0,0.0]
        v = vs[i]
        vt[0] = -v[1]
        vt[1] = v[0]
        fvt[0]= -vt[0]
        fvt[1]= -vt[1]
        fv[0] = -v[0]
        fv[1] = -v[1]
        
        p12 = [0,0]; p23 = [0,0]; p34 = [0,0]; p41 = [0,0]
        d1, d2, d3, d4 = [5, 5, 5, 5]
        e_min1 = 1000
        e_min2 = 1000
        e_min3 = 1000
        e_min4 = 1000
        active_d1 = True; active_d2 = True
        active_d3 = True; active_d4 = True
        iterations= True
        location_p1 = True; location_p2 = True
        location_p3 = True; location_p4 = True
        while (active_d1 or active_d2 or active_d3 or active_d4) and iterations:
            image2 = copy.copy(image)
            p12 = prpoint(seed_points[i],d1,d2,fvt,fv)
            p23 = prpoint(seed_points[i],d2,d3,fv,vt)
            p34 = prpoint(seed_points[i],d3,d4,vt,v)
            p41 = prpoint(seed_points[i],d4,d4,v,fvt)
            
            location_p1 = p12[0]<cols and p12[0]>0 and p12[1]<rows and p12[1]>0
            location_p2 = p23[0]<cols and p23[0]>0 and p23[1]<rows and p23[1]>0
            location_p3 = p34[0]<cols and p34[0]>0 and p34[1]<rows and p34[1]>0
            location_p4 = p41[0]<cols and p41[0]>0 and p41[1]<rows and p41[1]>0
            iterations  = location_p1 and location_p2 and location_p3 and location_p4
            
            e1,e2,e3,e4 = 0.0, 0.0, 0.0, 0.0
            e1 = power(image2,angle,p12,p23)
            e2 = power(image2,angle,p23,p34)
            e3 = power(image2,angle,p34,p41)
            e4 = power(image2,angle,p41,p12)
            
            zd = 5
            # @ active_d1
            if active_d1 == True:
                zp12 = zpoint(p12,fv,zd)
                zp23 = zpoint(p23,fv,zd)
                max_power = 0.0
                if power(image,angle,p12,zp12)<power(image,angle,p23,zp23):
                    max_power = power(image,angle,p12,zp12)
                else:
                    max_power = power(image,angle,p23,zp23)
                # Tg BetaTg LumTg =  a,b,c
                if abs(e1-90)<Tg and e_min1>abs(e1-90) and abs(max_power-90)>BetaTg:
                    active_d1 = False
                    e_min1    = abs(e1-90)
                else:
                    d2 = d2 + 1
            else:
                if abs(e1-90)>LumTg:
                    active_d1 = True
            # @active_d2
            if active_d2 == True:
                zp23 = zpoint(p23, vt, zd)
                zp34 = zpoint(p34, vt, zd)
                if power(image,angle,p23,zp23)<power(image,angle,p34,zp34):
                    max_power = power(image,angle,p23,zp23)
                else:
                    max_power = power(image,angle,p34,zp34)
                # Tg BetaTg LumTg =  a,b,c
                if abs(e2-90)<Tg and e_min2>abs(e2-90) and abs(max_power-90)>BetaTg:
                    active_d2 = False
                    e_min2    = abs(e2-90)
                else:
                    d3 = d3 + 1
            else:
                if abs(e2-90)>LumTg:
                    active_d2 = True
            # @active_d3
            if active_d3 == True:
                zp34 = zpoint(p34, v, zd)
                zp41 = zpoint(p41, v, zd)
                if power(image,angle,p34,zp34)<power(image,angle,p41,zp41):
                    max_power = power(image,angle,p34,zp34)
                else:
                    max_power = power(image,angle,p41,zp41)
                # Tg BetaTg LumTg =  a,b,c
                if abs(e3-90)<Tg and e_min3>abs(e3-90) and abs(max_power-90)>BetaTg:
                    active_d3 = False
                    e_min3    = abs(e3-90)
                else:
                    d4 = d4 + 1
            else:
                if abs(e3-90)>LumTg:
                    active_d3 = True
            # @active_d4
            if active_d4 == True:
                zp41 = zpoint(p41, fvt, zd)
                zp12 = zpoint(p12, fvt, zd)
                if power(image,angle,p41,zp41)<power(image,angle,p12,zp12):
                    max_power = power(image,angle,p41,zp41)
                else:
                    max_power = power(image,angle,p12,zp12)
                # Tg BetaTg LumTg =  a,b,c
                if abs(e4-90)<Tg and e_min4>abs(e4-90) and abs(max_power-90)>BetaTg:
                    active_d4 = False
                    e_min4    = abs(e4-90)
                else:
                    d4 = d4 + 1
            else:
                if abs(e4-90)>LumTg:
                    active_d4 = True
            
            cv2.line(image2,p12,p23,(0,255,255),2,8)
            cv2.line(image2,p23,p34,(0,255,255),2,8)
            cv2.line(image2,p34,p41,(0,255,255),2,8)
            cv2.line(image2,p41,p12,(0,255,255),2,8)
            
            cv2.waitKey(10)
            cv2.imshow("fdsfsdfasdf", image2)
        
        # end for while loop
        cv2.line(outimage,p12,p23,(0,255,255),2,8)
        cv2.line(outimage,p23,p34,(0,255,255),2,8)
        cv2.line(outimage,p34,p41,(0,255,255),2,8)
        cv2.line(outimage,p41,p12,(0,255,255),2,8)
        
        t = (float(cv2.getTickCount()) - t)/cv2.getTickFrequency()
        print "%d s.\n" % t 
    # end for for loop
    # return outimage
    cv2.imshow("sdfsfsdf", outimage)
    cv2.waitKey(0)
    # end for main

'''
if __name__ == "__main__":
    print ("Start run book spine segmentation by python!")
    print ("Edited by: JT")
    outimage = copy.copy(main())
    cv2.imshow("sdfsfsdf", outimage)
    cv2.waitKey(0)
'''