#!/usr/bin/python
# encoding:utf8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
# import const
# const.NOTAVALIDANGLE = 5
'''
@ var definition problem eg: a = 5; b = 10
@ for loop       problem eg: for i in range(start,end,step):
'''

Enum  = {'RETOK':0,'RETERROR':1}
PI2  = math.pi/2.0
CVPI = math.pi


NOTAVALIDANGLE = 5
MAXERROR         = 0.19625
ANGLEMARGIN      = 22.5

def ABS(a):
    if a < 0:
        return -a
    else:
        return a

def setTo14Quads(dp):
    if dp[1] < 0:
        dp[1] = -dp[1] # x
        dp[2] = -dp[2] # y

'''
typedef std::vector<cv::Point> LSEG;
'''
def DIRPOINT(pt, vx, vy):
    pt = pt
    x  = vx
    y  = vy
    return [pt, x, y]

'''
Constructor of class LSWMS (Slice Sampling Weighted Mean-Shift)
Args:
     -> imSize - Size of image
     -> R - accuracy parameter
     -> numMaxLSegs - requested number of line segments.
       if set to 0, the algorithm finds exploring
       the whole image until no more line segments can be found
     -> verbose - show messages
'''
class LSWMS:
    
    def __init__(self, imSize, R, numMaxLSegs, verbose):
        self.verbose     = verbose
        # Init variables            
        self.imSize      = imSize    # [height,width] = np.shape(Array)[0:2]
        self.imWidth     = imSize[1]
        self.imHeight    = imSize[0]
        self.R           = R            
        self.numMaxLSegs = numMaxLSegs       
        self.N           = 2*self.R + 1
        self.lSeg = []              # set of points need to be operated by np.array(lSeg)
        
        # Add padding if necessary 
        if (self.imSize[1] + 2*self.N) % 4 != 0:
            self.N = self.N + ((self.imSize[1] + 2*self.N)%4)/2
        self.imPadSize0  = self.imSize[0] + 2*self.N                    # self.imPadSize[0]  = self.imSize[0] + 2 * self.N
        self.imPadSize1  = self.imSize[1] + 2*self.N                    # self.imPadSize[1]  = self.imSize[1] + 2 * self.N
        self.imPadSize = [self.imPadSize0, self.imPadSize1]

        # Init images
        self.img     = np.zeros(self.imSize, np.typeNA['B'])             # np.typeNA['B'] : dtype = uint8
        self.imgPad  = np.zeros(self.imPadSize, np.typeNA['B'])
        self.roiRect = [self.N,self.N,self.imSize[1],self.imSize[0]]     # be careful

        # Mask image
        self.M = np.zeros(self.imPadSize,np.typeNA['B'])
        self.M.fill(255)

        # Angle mask
        self.A = np.zeros(self.imPadSize,np.float32)
        self.A.fill(NOTAVALIDANGLE)

        # Gradient images
        self.G  = np.zeros(self.imPadSize, np.typeNA['B'])
        self.Gx = np.zeros(self.imPadSize, np.float16) # np.float16
        self.Gy = np.zeros(self.imPadSize, np.float16) # np.float16

        # Iterator
        if self.numMaxLSegs != 0:
            self.sampleIterator = np.zeros((self.imSize[1]*self.imSize[0],1),np.typeNA['B'])
            for k in range(len(self.sampleIterator)):
                self.sampleIterator[k] = k
            # cv2.randShuffle(self.sampleIterator)
            np.random.shuffle(self.sampleIterator)

        # Angular margin
        self.margin = float(ANGLEMARGIN*CVPI/180)
    # end for init

    # def setitem(self, index, value):
    #     self.dict[index] = value

    '''
    **********************************************
    This function analyses the input image and finds
    line segments that are stored in the given vector
    Args:
     -> img    - Color or grayscale input image
     <- lSegs  - Output vector of line segments
     <- errors - Output vector of angular errors
    Ret:
     RETOK - no errors found
     RETERROR - errors found
    **********************************************
    '''
    def run(self, img, lSegs, errors ):
        # Clear line segment container
        lSegs = []   # set of Points lSegs.init
        errors= []   # set of double values

        # Input image to img
        if(img.shape[-1] == 3):
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            self.img = img

        # Add convolution borders
        # This way we avoid line segments at the boundaries of the image
        self.imgPad = cv2.copyMakeBorder(self.img, self.N, self.N, self.N, self.N, cv2.BORDER_REPLICATE)

        # Init Mask matrix
        self.M.fill(255) # 255
        np.array(self.M)
        # self.imgPadROI = self.M[self.roiRect[0]:self.roiRect[3],self.roiRect[1]:self.roiRect[2]]
        # self.imgPadROI.fill(0)
        for j in range(self.roiRect[0],self.roiRect[3]):
            for i in range(self.roiRect[1],self.roiRect[2]):
                self.M[j][i] = 0


        # Compute Gradient map
        # Call to the computation of the gradient and angle maps (SOBEL)
        retP, self.Gx,self.Gy,self.G = self.computeGradientMaps(self.imgPad, self.G, self.Gx, self.Gy)  # retP , self.Gx,self.Gy,self.G
        if retP == Enum['RETERROR']:
            if self.verbose:
                print ("ERROR: Probability map could not be computed\n")
            return  [Enum['RETERROR'],lSegs,errors]

        # Set padding to zeros
        NN = self.N + self.R
        self.Gx = self.setPaddingToZero(self.Gx, NN) # self.Gx
        self.Gy = self.setPaddingToZero(self.Gy, NN) # self.Gy
        self.G  = self.setPaddingToZero(self.G, NN)  # self.G

        # Line segment finder
        retLS, self.G,self.Gx,self.Gy, self.A,self.M, lSegs, errors = self.findLineSegments(self.G,self.Gx,self.Gy, self.A,self.M,lSegs,errors)
        return [retLS,lSegs,errors]

        # return Enum['RETOK']


    '''
    **********************************************
    SOBEL mode

    This function obtains the gradient image (G, Gx, Gy),
    and fills the angular map A.

    Args:
     -> img - Grayscale input image
     <- G - Gradient magnitude image
     <- Gx - Gradient x-magnitude image
     <- Gy - Gradient y-magnitude image
    Ret:
     RETOK - no errors found
     RETERROR - errors found
    **********************************************	
    '''
    def computeGradientMaps(self, img, G, Gx, Gy):

        if self.verbose:
            print ("Compute gradient maps...")
            #fflush(stdout);

        # Sobel operator
        ddepth = cv2.CV_16S
        afa    = float(1.0/8)
        Gx     = cv2.Sobel(img, ddepth, 1, 0)
        absGx  = cv2.convertScaleAbs(Gx,None,afa,0.0) # (double)1/8
        Gy     = cv2.Sobel(img, ddepth, 0, 1)
        absGy  = cv2.convertScaleAbs(Gy,None,afa,0.0)
        cv2.add(absGx,absGy,G)


        # meanG        = list[cv2.mean(G)]  # cv::Scalar
        self.meanG = int(cv2.mean(G)[0])   # self.meanG = int (meanG.val[0])

        if self.verbose:
            print ("computed: meanG = %d\n" % self.meanG)

        movedCounter = 0
        for j in range(self.imPadSize[0]):
            for i in range(self.imPadSize[1]):
                if Gx[j,i] < 0:
                    Gy[j,i] = -Gy[j,i]
                    Gx[j,i] = -Gx[j,i]
                    movedCounter = movedCounter + 1

        if self.verbose:
            print ("Moved %d , %d , %.2f elements to 1st4th quadrant" % (movedCounter, self.imPadSize[0]*self.imPadSize[1], (100.0*movedCounter)/(self.imPadSize[0]*self.imPadSize[1])))

        if self.meanG > 1 and  self.meanG < 256:
            return [Enum['RETOK'], Gx, Gy, G]       # [Enum['RETOK'], Gx, Gy, G]

        else:
            return [Enum['RETERROR'], Gx, Gy, G]    # [Enum['RETERROR'], Gx, Gy, G]
    # end for computeGradientMaps	



    '''
    **********************************************
    This function finds line segments using the 
    probability map P, the gradient components 
    Gx, and Gy and the angle map A.

    Args:
     -> G - Gradient magnitude map
     -> Gx - Gradient x-magnitude image
     -> Gy - Gradient y-magnitude image
     <- M - Mask image of visited pixels
     <- lSegs - vector of detected line segments
     <- errors - vector of angular errors
    Ret:
     RETOK - no errors found
     RETERROR - errors found
    **********************************************	
    '''
    def findLineSegments(self, G, Gx, Gy, A, M, lSegs, errors):
        # Loop over the image
        x0, y0, kInterator = 0, 0, 0
        rows = np.shape(self.img)[0]
        cols = np.shape(self.img)[1]
        imgSize = cols * rows

        while True:
            if kInterator == imgSize:
                break
            if self.numMaxLSegs == 0:
                x0 = kInterator%cols
                y0 = kInterator/cols
            else:
                x0 = self.sampleIterator[kInterator]%cols
                y0 = self.sampleIterator[kInterator]/cols
            kInterator = kInterator + 1

            # Add padding
            x0 = x0 + self.N
            y0 = y0 + self.N

            # Check mask and value
            Mval = self.M[y0,x0]
            Gval = self.G[y0,x0]
            Threh = self.meanG

            if (Mval == 0):                  # if (self.M[y0,x0] == 0) and (G[y0,x0] > self.meanG):
                ptOrig = (x0,y0)                                  # !The sample is (x0,y0)
                gX = float(Gx[y0,x0])
                gY = float(Gy[y0,x0])
                # Since it is computed from Gx, Gy, it is in 1°4°
                dpOrig = DIRPOINT(ptOrig, gX, gY)  # dpOrig = DIRPOINT(ptOrig, gX, gY)

                # Line segment generation
                error = 0
                if self.verbose:
                    print ("-------------------------------\n")
                if self.verbose:
                # % dpOrig.pt.x, dpOrig.pt.y, dpOrig.vx, dpOrig.vy
                    print ("Try dpOrig=(%d,%d,%.2f,%.2f)...\n" % (dpOrig[0][0], dpOrig[0][1], dpOrig[0], dpOrig[1]))
                retLS, dpOrig, self.lSeg, error = self.lineSegmentGeneration(dpOrig, self.lSeg, error)  # self.lSeg is set of points - - - - - - >
                
                if retLS == Enum['RETOK'] and error < MAXERROR:
                    if self.verbose:
                        #  % lSeg[0].x, lSeg[0].y, lSeg[1].x, lSeg[1].y
                        print ("lSeg generated=(%d,%d)->(%d,%d)...\n" % (self.lSeg[0][0], self.lSeg[0][1], self.lSeg[1][0], self.lSeg[1][1]))
                    if self.verbose:
                        print "-------------------------------\n"
                    lSegs.append(self.lSeg)
                    errors.append(1.0*error)

                    if self.numMaxLSegs != 0 and len(lSegs) >= int(self.numMaxLSegs):
                        break
                else:
                    # Mark as visited
                    w = [x0-self.R, y0-self.R, self.N, self.N]
                    # roi = self.M[w[0]:w[3],w[1]:w[2]]
                    # roi.fill(255)
                    for j in range(w[0],w[3]):
                        for i in range(w[1],w[2]):
                            self.M[j,i] = 255

        return [Enum['RETOK'], G, Gx, Gy, A, M, lSegs, errors]
    # end for findLineSegments


    '''
    **********************************************
    Starts at dpOrig and generates lSeg

    Args:
     -> dpOrig - starting DIRPOINT
     <- lSeg - detected line segment
    Ret:
     RETOK - lSeg created
     RETERROR - lSeg not created
    **********************************************
    '''
    def lineSegmentGeneration(self, dpOrig, lSeg, error):
        # Check input data
        rows = np.shape(self.G)[0]
        cols = np.shape(self.G)[1]
        # dpOrig.pt.x , dpOrig.pt.y
        if dpOrig[0][0] < 0 or dpOrig[0][0] >= cols or dpOrig[0][1] < 0 or dpOrig[0][1] >= rows:
            return [Enum['RETERROR'],dpOrig, lSeg, error]
        
        # Find best candidate with Mean-Shift
        dpCentr = copy.copy(dpOrig)           # copy.copy
        if self.verbose:
            # % dpOrig.pt.x, dpOrig.pt.y, dpOrig.vx, dpOrig.vy
            print ("\tMean-Shift(Centr): from (%d,%d,%.2f,%.2f) to..." % (dpOrig[0][0], dpOrig[0][1], dpOrig[1], dpOrig[2]))
            #fflush(stdout)
        # COMO LE PASO M, TIENE EN CUENTA SI SE HA VISITADO O NO
        retMSC, dpOrig, dpCentr, self.M = self.weightedMeanShift(dpOrig, dpCentr, self.M)  # !< function of weightedMeanShift - - - - - - >
        if self.verbose:
            # % dpCentr.pt.x, dpCentr.pt.y, dpCentr.vx, dpCentr.vy
            print ("(%d,%d,%.2f, %.2f)\n" % (dpCentr[0][0], dpCentr[0][1], dpCentr[1], dpCentr[2]))
        if retMSC == Enum['RETERROR']:
            if self.verbose:
                print "\tMean-Shift reached not a valid point\n"
        return [Enum['RETERROR'],dpOrig, lSeg, error]


        # Grow in two directions from dpCentr
        #if self.verbose:
        #    print ("\t GROW 1:")
        #    #fflush(stdout)

        pt1 = [0,0]
        retG1,dpCentr,pt1,dir1 = grow(dpCentr, pt1, 1)          # !!!< function of grow
        # (float)((dpCentr.pt.x - pt1.x)*(dpCentr.pt.x - pt1.x) + (dpCentr.pt.y - pt1.y)*(dpCentr.pt.y - pt1.y))
        d1    = (float)((dpCentr[0][0] - pt1[0])*(dpCentr[0][0] - pt1[0]) + (dpCentr[0][1] - pt1[1])*(dpCentr[0][1] - pt1[1]))
        if self.verbose:
            # % pt1.x, pt1.y
            print ("\tpt1(%d,%d), dist = %.2f, error=%.4f\n" % (pt1[0], pt1[1], d1, retG1))

        if self.verbose:
            print ("\tGROW 2:")
            #fflush(stdout)
        pt2   = [0,0]
        retG2,dpCentr,pt2,dir2 = grow(dpCentr, pt2, 2)         # !!!< function of grow
        d2    = float((dpCentr[0][0] - pt2[0])*(dpCentr[0][0] - pt2[0]) + (dpCentr[0][1] - pt2[1])*(dpCentr[0][1] - pt2[1]))
        if self.verbose:
            print ("\tpt1(%d,%d), dist = %.2f, error=%.4f\n" % (pt2[0], pt2[1], d2, retG2))

        if retG1 == -1 and retG2 == -1:
            return [Enum['RETERROR'],dpOrig, lSeg, error]

        # Select the most distant extremum
        if d1 < d2:
            pt1 = pt2
            error = retG2
            if self.verbose:
                print ("Longest dir is 2 \n")
        else:
            error = retG1
            if self.verbose:
                print ("Longest dir is 1 \n")

        # Grow to the non-selected direction, with the new orientation
        dirX = float(dpCentr[0][0] - pt1[0])  # dpCentr.pt.x - pt1.x
        dirY = float(dpCentr[0][1] - pt1[1])  # dpCentr.pt.y - pt1.y
        norm = math.sqrt(dirX*dirX + dirY*dirY)

        if norm > 0:
            dirX = dirX/norm
            dirY = dirY/norm
            # DIRPOINT must be filled ALWAYS with gradient vectors
            dpAux = DIRPOINT(dpCentr[0], -(-dirY), -dirX) # dpAux = DIRPOINT(dpCentr[0], -(-dirY), -dirX)
            retG,dpAux, pt2, dir = 1.0*grow(dpAux, pt2, 1)     # !!!< function of grow
            error = retG
        else:
            pt2 = dpCentr[0]                   # dpCentr.pt

        # Check
        dirX = float(pt1[0] -pt2[0])         # pt1.x -pt2.x
        dirY = float(pt1[1] -pt2[1])         # pt1.y -pt2.y
        if math.sqrt(dirX*dirX + dirY*dirY) < self.N:
            if self.verbose:
                print ("Line segment not generated: Too short.\n")
                return [Enum['RETERROR'],dpOrig, lSeg, error]
        
        # Output line segment
        if self.verbose:
            # % pt2.x, pt2.y, pt1.x, pt1.y
            print ("LSeg = (%d,%d)-(%d,%d)\n" % (pt2[0], pt2[1], pt1[0], pt1[1]))
        lSeg = list()
        lSeg.append((pt2[0] - 2*self.R, pt2[1] - 2*self.R))
        lSeg.append((pt1[0] - 2*self.R, pt1[1] - 2*self.R))

        # Update visited positions matrix
        self.updateMask(pt1,pt2)          # !< function of updateMask - - Do not need return
        return [Enum['RETOK'],dpOrig, lSeg, error]
    # end for lineSegmentGeneration


    '''
    @updateMask-> Bresenham from one extremum to the other
    '''
    def updateMask(self, pt1, pt2):
        # x1 = pt1.x, x2 = pt2.x, y1 = pt1.y, y2 = pt2.y
        x1 = pt1[0]
        x2 = pt2[0]
        y1 = pt1[1]
        y2 = pt2[1]

        dx = ABS(x2-x1)
        dy = ABS(y2-y1)

        sx,sy,err,e2 = [0,0,0,0]

        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if  y1 < y2:
            sy  = 1
        else:
            sy = -1
        err = dx - dy

        while True: #Current value is (x1, y1) # DO... Set window to "visited = 255"
            for j in range(y1-self.R,y1+self.R+1):
                for i in range(x1-self.R,x1+self.R+1):
                    self.M[j,i] = 255

            # Check end
            if x1 == x2 and y1 == y2:
                 break

            # Update position for next iteration
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x1  = x1  + sx
            if e2 < dx:
                err = err + dx
                y1  = y1  + sy
    # end for updateMask



    '''
    **********************************************
    Refines dpOrig and creates dpDst

    Args:
     -> dpOrig - starting DIRPOINT
     <- dpDst - refined DIRPOINT
    Ret:
     RETOK - dpDst created
     RETERROR - dpDst not found

    Called from "lineSegmentGeneration"
    *********************************************
    '''
    def weightedMeanShift(self, dpOrig, dpDst, M):
        # MAIN LOOP: loop until MS generate no movement (or dead-loop)
        self.seeds = []
        dpCurr = copy.copy(dpOrig)       # The initial dp is in 1°4°
        dpDst  = copy.copy(dpOrig)

        while True:
            # Check point
            # dpCurr.pt.x, dpCurr.pt.y
            rows = np.shape(self.G)[0]
            cols = np.shape(self.G)[1]

            if dpCurr[0][0]<0 or dpCurr[0][0]>=cols or dpCurr[0][1]<0 or dpCurr[0][1]>=rows:
                return [Enum['RETERROR'], dpOrig, dpDst, M]

             # Check direction
             # dpCurr.vx, dpCurr.vy
            if dpCurr[1] == 0 and dpCurr[2] == 0:
                return [Enum['RETERROR'], dpOrig, dpDst, M]

            # Convert to 1°4° (maybe not needed)
            setTo14Quads(dpCurr)

            # Check already visited
            if len(M)!=0:
                # M.at<uchar>(dpCurr.pt.y, dpCurr.pt.x) == 255
                if M[dpCurr[0][1],dpCurr[0][0]]==255:
                    return [Enum['RETERROR'], dpOrig, dpDst, M]

            # Check if previously used as need for this MS-central (this is to avoid dead-loops)
            for i in range(len(self.seeds)):
                # seeds[i].x == dpCurr.pt.x && seeds[i].y == dpCurr.pt.y
                if self.seeds[i][0] == dpCurr[0][0] and self.seeds[i][1] == dpCurr[0][1]:
                    dpDst = copy.copy(dpCurr)
                    return [Enum['RETERROR'], dpOrig, dpDst, M]
            
             # Define bounds
            xMin = dpCurr[0][0] - self.R
            yMin = dpCurr[0][1] - self.R
            xMax = dpCurr[0][0] + self.R
            yMax = dpCurr[0][1] + self.R
            offX = self.R
            offY = self.R
            
            if xMin<0 or yMin<0 or xMax>=cols or yMax>=rows:
                return [Enum['RETERROR'], dpOrig, dpDst, M]
            
            self.seeds.append(dpCurr[0])
            
            # !!!< Define rois be careful
            xy  = [xMin,yMin]
            WH  = [xMax-xMin+1,yMax-yMin+1]
            roi = [xy, WH]
            gBlock  = self.G[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            gXBlock = self.Gx[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            gYBlock = self.Gy[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            aBlock  = self.A[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            insideBlock = np.ones(np.shape(gBlock),np.typeNA['B'])
            
            # Update angles (this is to compute angles only once)
            RC = np.shape(aBlock)[0:2]
            
            for j in range(0,RC[0]):
                for i in range(0,RC[1]):
                    aBlock[j,i] = math.atan2(1.0*gYBlock[j,i],1.0*gXBlock[j,i])
            
            # Angle analysis output is between (-CVPI/2, CVPI/2)
            currentAngle = 1.0*math.atan2(dpCurr[2],dpCurr[1])
            angleShift     = 0.0
            outsideCounter = 0
            if currentAngle - self.margin < -PI2:
                # Shift angles according to currentAngle to avoid discontinuities
                # if(verbose) printf("shift angles since %.2f - %.2f < %.2f\n", currentAngle, margin, -PI2);
                angleShift   = currentAngle
                aBlock       = aBlock - currentAngle
                currentAngle = 0
                minAngle     = float(currentAngle - self.margin)
                maxAngle     = float(currentAngle + self.margin)

                for j in range(0, RC[0]):
                    for i in range(0, RC[1]):
                        if aBlock[j,i] < -PI2:
                            aBlock[j,i] = aBlock[j,i] + 1.0*CVPI
                        if aBlock[j,i] >  PI2:
                            aBlock[j,i] = aBlock[j,i] - 1.0*CVPI
                        if aBlock[j,i] < minAngle or aBlock[j,i] > maxAngle:
                            insideBlock[j,i] = 0
                            outsideCounter   = outsideCounter + 1
                # Restore
                aBlock = aBlock + angleShift
            
            elif currentAngle + self.margin > PI2:
                # Shift angles according to currentAngle to avoid discontinuities
                # if(verbose) printf("shift angles since %.2f + %.2f > %.2f\n", currentAngle, margin, PI2);
                angleShift   = currentAngle
                aBlock       = aBlock - currentAngle
                currentAngle = 0
                
                minAngle = float(currentAngle - self.margin)
                maxAngle = float(currentAngle + self.margin)
                
                for j in range(0, RC[0]):
                    for i in range(0, RC[1]):
                        if aBlock[j,i] < -PI2:
                            aBlock[j,i] = aBlock[j,i] + 1.0*CVPI
                        if aBlock[j,i] >  PI2:
                            aBlock[j,i] = aBlock[j,i] - 1.0*CVPI
                        if aBlock[j,i] < minAngle or aBlock[j,i] > maxAngle:
                            insideBlock[j,i] = 0
                            outsideCounter   = outsideCounter + 1
                # Restore
                aBlock = aBlock + angleShift

            else:
                angleShift = 0
                minAngle   = float(currentAngle - self.margin)
                maxAngle   = float(currentAngle + self.margin)

                for j in range(0, RC[0]):
                    for i in range(0, RC[1]):
                        if aBlock[j,i]<minAngle or aBlock[j,i]>maxAngle:
                            insideBlock[j,i] = 0
                            outsideCounter   = outsideCounter + 1

            # Check number of samples inside the bandwidth
            if outsideCounter == (2*self.R+1)*(2*self.R+1):
                return [Enum['RETERROR'], dpOrig, dpDst, M]

            # New (Circular) Mean angle (weighted by G)
            sumWeight = 0.0; foffX = 0.0; foffY = 0.0; meanAngle = 0.0

            RCG = np.shape(gBlock)[0:2]

            for j in [0,RCG[0]]:
                for i in [0,RCG[1]]:
                    if insideBlock[j,i] != 0:
                        # This sample is inside the Mean-Shift bandwidth
                        # Weighted mean of positons
                        foffX = foffX + 1.0*(i+1)*gBlock[j,i] # // This cannot be precomputed...
                        foffY = foffY + 1.0*(j+1)*gBlock[j,i]

                        # Weighted mean of angle
                        meanAngle = meanAngle + aBlock[j,i]*gBlock[j,i]
                        sumWeight = sumWeight + gBlock[j,i]

            foffX /= sumWeight
            foffX = foffX - 1
            foffY /= sumWeight
            foffY = foffY - 1
            meanAngle /= sumWeight

            # Check convergence (movement with respect to the center)
            if int(foffX) == offX and int(foffY) == offY:
                dpDst = DIRPOINT(dpCurr[0], math.cos(meanAngle), math.sin(meanAngle)) # dpDst = DIRPOINT(dpCurr[0], math.cos(meanAngle), math.sin(meanAngle))
                setTo14Quads(dpDst)
                return [Enum['RETOK'], dpOrig, dpDst, M]
            else:
                # Not converged: update dpCurr and iterate
                dpCurr[0][0] = dpCurr[0][0] + int(foffX) - offX
                dpCurr[0][1] = dpCurr[0][1] + int(foffY) - offY
                dpCurr[1]    = math.cos(meanAngle)
                dpCurr[2]    = math.sin(meanAngle)

        return [Enum['RETOK'], dpOrig, dpDst, M]
    # end for weightedMeanShift


    '''
    **********************************************
    Finds end-point ptDst starting from dpOrig 
    Args:
        -> dpOrig - starting DIRPOINT 
        <- ptDst - end-point
        -> dir - growing direction (1(+) or 2(-))
    Ret:
        error - error of line segment
        Called from lineSegmentGeneration
    **********************************************	
    '''
    def grow(self, dpOrig, ptDst, dir):
        # auxiliar
        ptEnd1 = [0,0]; ptEnd2 = [0,0]
        dpEnd  = [[0,0],0,0]
        dpRef  = [[0,0], 0, 0]  # dpEnd  = DIRPOINT([[0,0],0,0]); dpRef  = DIRPOINT( [[0,0], 0, 0])
        
        # Init output
        ptDst = dpOrig[0]
        
        # Starting gradient vector and director vector
        gX = 0.0; gY = 0.0
        if dir == 1:
            gX = dpOrig[1]
            gY = dpOrig[2]
        elif dir == 2:
            gX = -dpOrig[1]
            gY = -dpOrig[2]
        else:
            return [Enum['RETERROR'],dpOrig, ptDst, dir]
            
        # Compute currentAngle in 1°4°
        error1 = 0.0
        auxAngle  = 0.0
        minAngle  = 0.0
        maxAngle  = 0.0
        diffAngle = 0.0
        growAngle = math.atan2(gY,gX)
        
        # Starting point and angle - Bresenham
        pt1 = dpOrig[0]
        pt2 = [pt1[0] + int(1000*(-gY)),pt1[1] + int(1000*(gX))]
        cv2.clipLine(self.imPadSize, pt1, pt2)                   # !be careful
        
        # Loop - Bresenham
        k1 = 0
        x1 = pt1[0], x2 = pt2[0]
        y1 = pt1[1], y2 = pt2[1]
        dx = ABS(x2-x1)
        dy = ABS(y2-y1)
        sx = 0; sy = 0; err = 0; e2 = 0
        
        if self.verbose:
            print ("From (%d,%d) to (%d,%d)..." %  (x1, y1, x2, y2))
            # fflush(stdout)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        err = dx - dy
        
        maxNumZeroPixels = 2 * self.R; countZeroPixels = 0
        while True:
            '''
            Current value is (x1,y1)	
            if(verbose) { printf("\n\tBresenham(%d,%d)", x1, y1); fflush(stdout); }
            -------------------------------
            Do...
            Check if angle has been computed
            '''
            if self.A[y1,x1] != NOTAVALIDANGLE:
                auxAngle = self.A[y1,x1]
            else:
                auxAngle = math.atan2(1.0*self.Gy[y1,x1], 1.0*self.Gx[y1,x1])
                self.A[y1,x1] = auxAngle
            # Check early-termination of Bresenham
            if self.G[y1,x1] == 0:
                countZeroPixels = countZeroPixels + 1
                if countZeroPixels >= maxNumZeroPixels:
                    break
            
            # Check angular limits
            if growAngle - self.margin < -PI2:
                # e.g. currentAngle = -80°, margin = 20°
                minAngle = growAngle - self.margin + 1.0*CVPI #  e.g. -80 -20 +180 = 80
                maxAngle = growAngle + self.margin             #  e.g. -80 +20
                
                if auxAngle < 0:
                    if auxAngle > maxAngle:
                        break
                    diffAngle = ABS(growAngle - auxAngle)
                else:
                    if auxAngle < minAngle:
                        break
                    diffAngle = ABS(growAngle - (auxAngle - 1.0*CVPI))
            elif growAngle + self.margin > PI2:
                # e.g. currentAngle = 80º, margin = 20º
                minAngle = growAngle - self.margin             # e.g. 80 - 20 = 60
                maxAngle = growAngle + self.margin - 1.0*CVPI # e.g. 80 +20 -180 = -80
                
                if auxAngle > 0:
                    if auxAngle < minAngle:
                        break
                    diffAngle = ABS(growAngle - auxAngle)
                else:
                    if auxAngle > maxAngle:
                        break
                    diffAngle = ABS(growAngle - (auxAngle + 1.0*CVPI))
            else:
                minAngle = growAngle - self.margin
                maxAngle = growAngle + self.margin
                if auxAngle < minAngle or auxAngle > maxAngle:
                    break
                diffAngle = ABS(growAngle - auxAngle)        
            error1 = error1 + diffAngle
            ptEnd1 = [x1, y1]
            k1 = k1 + 1
            
            # Check end
            if x1 == x2 and y1 == y2:
                break
            # Update position for next iteration
            e2 = 2*err
            if e2 > -dy:
                err = err - dy
                x1  = x1 + sx
            if e2 < dx:
                err = err + dx
                y1  = y1  + sy
            
        # "k1": how many pints have been visited
        # "ptEnd": last valid point
        if k1 == 0:
            # this means that even the closest point has not been accepted	
            ptEnd1 = dpOrig[0]
            error1 = 1.0*CVPI
        else:
            error1 /= k1
        if self.verbose:
            print (", Arrived to (%d,%d), error=%.2f" % (ptEnd1[0],ptEnd1[1],error1))
            # fflush(stdout)
        # Set ptDst
        ptDst = copy.copy(ptEnd1)
        
        # Apply Mean-Shift to refine the end point
        if self.verbose:
            print (", Dist = (%d,%d)\n" % (ABS(ptEnd1[0] - dpOrig[0][0]), ABS(ptEnd1[1] - dpOrig[0][1])))
        if ABS(ptEnd1[0] - dpOrig[0][0]) > self.R or ABS(ptEnd1[1] - dpOrig[0][1]) > self.R:
            # ONLY IF THERE HAS BEEN (SIGNIFICANT) MOTION FROM PREVIOUS MEAN-SHIFT MAXIMA
            counter = 0
            while True:
                if self.verbose:
                    print ("\tMean-Shift(Ext): from (%d,%d,%.2f,%.2f) to..." % (ptEnd1[0], ptEnd1[1], gX, gY))
                    # fflush(stdout)
                counter = counter + 1
                # Mean-Shift on the initial extremum
                dpEnd[0] = copy.copy(ptEnd1) 
                dpEnd[1] = copy.copy(gX) 
                dpEnd[2] = copy.copy(gY)
                
                dpRef[0] = copy.copy(ptEnd1)
                dpRef[1] = copy.copy(gX)
                dpRef[2] = copy.copy(gY)
                retMSExt, dpEnd, dpRef,self.M = self.weightedMeanShift(dpEnd, dpRef,self.M)   # !< function of weightedMeanShift
                
                if self.verbose:
                    print ("(%d,%d,%.2f,%.2f)\n" % (dpRef[0][0],dpRef[0][1], dpRef[1], dpRef[2]))
                if retMSExt == Enum['RETERROR']:
                    ptDst = copy.copy(ptEnd1)
                    return [Enum['RETOK'],dpOrig, ptDst, dir]
                
                # Check motion caused by Mean-Shift
                if (dpRef[0][0] == dpEnd[0][0]) and (dpRef[0][1] == dpEnd[0][1]):
                    ptDst = copy.copy(dpRef[0])
                    return [Enum['RETOK'],dpOrig, ptDst, dir]
                
                # Check displacement from dpOrig
                gX = float(dpRef[0][1] - dpOrig[0][1])
                gY = float(dpOrig[0][0] - dpRef[0][0])
                if gX == 0 and gY == 0:
                    ptDst = copy.copy(dpRef[0])
                    return [Enum['RETOK'],dpOrig, ptDst, dir]
                norm = 1.0*math.sqrt(gX*gX + gY*gY)
                gX /= norm
                gY /= norm
                
                # New Bresenham procedure
                if gX < 0:
                    # Move to 1°4°
                    gX = -gX
                    gY = -gY
                growAngle = math.atan2(gY,gX)
                
                k2 = 0; error2 = 0.0
                
                pt2[0] = pt1[0] + int(1000*(-gY))
                pt2[1] = pt1[1] + int(1000*gX)
                
                x1 = copy.copy(pt1[0]); x2 = copy.copy(pt2[0])
                y1 = copy.copy(pt1[1]); y2 = copy.copy(pt2[1])
                dx = ABS(x2-x1)
                dy = ABS(y2-y1)
                
                if x1 < x2:
                    sx = 1
                else:
                    sx = -1
                if y1 < y2:
                    sy = 1
                else:
                    sy = -1
                err = dx - dy
                
                if self.verbose:
                    print ("\tRefined GROW: From (%d,%d) to (%d,%d)..." % x1, y1, x2, y2)
                    # fflush(stdout)
                while True:
                    if self.A[y1,x1] != NOTAVALIDANGLE:
                        auxAngle = self.A[y1,x1]
                    else:
                        auxAngle = math.atan2(1.0*self.Gy[y1,x1],1.0*self.Gx[y1,x1])
                        self.A[y1,x1] = auxAngle
                    
                    # Check early-termination of Bresenham
                    if self.G[y1,x1] == 0:
                        countZeroPixels = countZeroPixels + 1
                        if countZeroPixels >= maxNumZeroPixels:
                            break  # No gradient point
                            
                    # Check angular limits
                    if growAngle - self.margin < -PI2:
                        minAngle = growAngle - self.margin + 1.0*CVPI
                        maxAngle = growAngle + self.margin
                        
                        if auxAngle < 0:
                            if auxAngle > maxAngle:
                                break
                            diffAngle = ABS(growAngle - auxAngle)
                        else:
                            if auxAngle < minAngle:
                                break
                            diffAngle = ABS(growAngle - (auxAngle - 1.0*CVPI))
                    
                    elif growAngle + self.margin > PI2:
                        minAngle = growAngle - self.margin
                        maxAngle = growAngle + self.margin - 1.0*CVPI
                        
                        if auxAngle > 0:
                            if auxAngle < minAngle:
                                break
                            diffAngle = ABS(growAngle - auxAngle)
                        else:
                            if auxAngle > maxAngle:
                                break
                            diffAngle = ABS(growAngle - (auxAngle + 1.0*CVPI))
                    
                    else:
                       minAngle = growAngle - self.margin
                       maxAngle = growAngle + self.margin
                       if auxAngle < minAngle or auxAngle > maxAngle:
                           break
                       diffAngle = ABS(growAngle - auxAngle)
                   
                    error2 = error2 + diffAngle
                    ptEnd2 = [x1,y1]
                    k2 = k2 + 1
                    
                    # Check end
                    if x1 == x2 and y1 == y2:
                        break
                    # Update position for next iteration
                    e2 = 2*err
                    if e2 > -dy:
                        err = err - dy
                        x1  = x1  + sx
                    if e2 < dx:
                        err = err + dx
                        y1  = y1  +  sy
                # Bresenham while
                
                # "k2": how many points have been visited
                # "ptEnd2": last valid point
                if k2 == 0:
                    # this means that even the closest point has not been accepted	
                    ptEnd2 = copy.copy(dpOrig[0])
                    error2 = 1.0*CVPI
                else:
                    error2 = error2 / k2
                    # fflush(stdout) # Don't really know why, but this is necessary to avoid dead loops...
                
                if self.verbose:
                    print (", Arrived to (%d,%d), error=%.2f" % (ptEnd2[0], ptEnd2[1], error2))
                    # fflush(stdout)
                if self.verbose:
                    print (", Dist = (%d,%d)\n" % (ABS(ptEnd2[0] - dpOrig[0][0]), ABS(ptEnd1[1] - dpOrig[0][1])))
                        
                # Compare obtained samples
                if error1 <= error2:
                    ptDst = copy.copy(ptEnd1)
                    return error1
                else:
                    ptEnd1 = copy.copy(ptEnd2)
                    k1     = copy.copy(k2)
                    error1 = copy.copy(error2)
                
            # Mean-Shift while
        return [error1,dpOrig, ptDst, dir]
    # end for grow


    '''
    @setPaddingToZero
    '''            
    def setPaddingToZero(self, img, NN):
        H,W,= np.shape(img)[0:2]
        cv2.rectangle(img, (0,0),    (W-1,NN-1),  (0,0,0), 1)
        cv2.rectangle(img, (0,0),    (NN-1, H-1), (0,0,0), 1)
        cv2.rectangle(img, (0,H-NN), (W-1,H-1),   (0,0,0), 1)
        cv2.rectangle(img, (W-NN,0), (W-1,H-1),   (0,0,0), 1)
        return img
    # end for setPaddingToZero

    '''
    @drawLSegs

    def drawLSegs(self, img, lSegs, color, thickness):
        for i in range(0,len(lSegs)):
            cv2.line(img, lSegs[i][0], lSegs[i][1], color, thickness)
    '''

    def drawLSegs(self, img, lSegs, errors, thickness):
        colors = [(255,0,0),(200,0,0),(150,0,0),(50,0,0)]
        for i in range(0,len(lSegs)):
            if errors[i] < 0.087:    # 5° 
                cv2.line(img, lSegs[i][0], lSegs[i][1], colors[0], 1)
            elif errors[i] < 0.174:  # 10°
                cv2.line(img, lSegs[i][0], lSegs[i][1], colors[1], 1)
            elif errors[i] < 0.26:   # 15°
                cv2.line(img, lSegs[i][0], lSegs[i][1], colors[2], 1)
            else:
                cv2.line(img, lSegs[i][0], lSegs[i][1], colors[3], 1)
        return img
    # end for drasLSegs




