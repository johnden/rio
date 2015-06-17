//
//  line_segment.h
//  opencv
//
//  Created by incer on 15/5/7.
//  Copyright (c) 2015年 ce. All rights reserved.
//

#ifndef __opencv__line_segment__
#define __opencv__line_segment__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "LSWMS.h"

using namespace cv;
using namespace std;

class ls
{
private:
    int krows = 640;
    float la =30,lb = 70,lc = 80;
    double line_size(Point &p1,Point &p2);
    float line_jiao(Point &p1,Point &p2);
    int ca(LSEG &line,int k);
    void run(vector<LSEG> &lines1,int left,int right,int k);
    void QuickSort(vector<LSEG> &lines1,int Count,int k);
    double power(Mat &src,Mat &angle,Point &a,Point &b);
    Point prpoint(Point &center,int d,Point2f &v);
    int point_line(Point &p1,Point &p2,Point &tp);
    int warf(Mat &src,Mat &src1,Mat &angle,LSEG &line1,LSEG &line2,Mat &quad ,int k);
    bool pr_detect(Mat &src,Mat &src1,Mat &angle,vector<LSEG> &oolines,vector<Mat> &result,int k);
    void findline(Mat &flood,Mat &angle,vector<LSEG> &lines,int k);
    void parallelines(vector<LSEG> &lines,vector<LSEG> &outlines);
public:
    void bookSegmentStart(Mat &src1,vector<Mat> &result);
};
#endif /* defined(__opencv__line_segment__) */
