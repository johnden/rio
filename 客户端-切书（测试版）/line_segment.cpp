//
//  line_segment.cpp
//  opencv
//
//  Created by incer on 15/5/7.
//  Copyright (c) 2015年 ce. All rights reserved.
//

#include "line_segment.h"
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "LSWMS.h"
//-----------------------------------【命名空间声明部分】---------------------------------------
//	     描述：包含程序所使用的命名空间
//-----------------------------------------------------------------------------------------------
using namespace cv;
using namespace std;
//-----------------------------------【main( )函数】--------------------------------------------
//	     描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------

double ls::line_size(Point &p1,Point &p2)
{
    return sqrt(pow(p1.x - p2.x,2)+pow(p1.y - p2.y,2));
}

float ls::line_jiao(Point &p1,Point &p2)
{
    if(p2.y == p1.y)
        return 0;
    else
    {
        Point min = p2.y<p1.y?p2:p1;
        Point max = p2.y>p1.y?p2:p1;
        return acos((max.x - min.x)/line_size(p1, p2))*(180.0/CV_PI);
    }
}


int ls::ca(LSEG &line,int k)
{
    float kdx = (float)line[0].x - (float)line[1].x;
    float kdy = (float)line[0].y - (float)line[1].y;
    Point zp2;
    zp2.x = (line[0].x + line[1].x)/2.0;
    zp2.y = (line[0].y + line[1].y)/2.0;
    if(k==0)
    {
        return (line[0].x+line[1].x)*0.5;
    }
    else if(k==1)
    {
        return (line[0].y+line[1].y)*0.5;
    }
    else if(k==2)
    {
        return abs(- (kdx/line_size(line[0],line[1]))*krows + (kdx/line_size(line[0],line[1]))*zp2.y -(kdy/line_size(line[0],line[1]))*zp2.x);
    }
    else if(k==3)
    {
        return abs((kdx/line_size(line[0],line[1]))*zp2.y - (kdy/line_size(line[0],line[1]))*zp2.x);
    }
    else
    {
        return line_jiao(line[0],line[1]);
    }
}

void ls::run(vector<LSEG> &lines1,int left,int right,int k)
{
    int i,j;
    int middle;
    LSEG iTemp;
    i = left;
    j = right;
    middle = ca(lines1[(left+right)/2],k); //求中间值
    do{
        while((ca(lines1[i],k)<middle) && (i<right))//从左扫描大于中值的数
            i++;
        while((ca(lines1[j],k)>middle) && (j>left))//从右扫描大于中值的数
            j--;
        if(i<=j)//找到了一对值
        {
            //交换
            iTemp = lines1[i];
            lines1[i] = lines1[j];
            lines1[j] = iTemp;
            i++;
            j--;
        }
    }while(i<=j);//如果两边扫描的下标交错，就停止（完成一次）
    //当左边部分有值(left<j)，递归左半边
    if(left<j)
        run(lines1,left,j,k);
    //当右边部分有值(right>i)，递归右半边
    if(right>i)
        run(lines1,i,right,k);
}

void ls::QuickSort(vector<LSEG> &lines1,int Count,int k)
{
    run(lines1, 0,Count-1,k);
}

double ls::power(Mat &src,Mat &angle,Point &a,Point &b)
{
    
    double dx = (b.x-a.x)/sqrt(pow(b.x-a.x,2)+pow(b.y-a.y, 2));
    double dy = (b.y-a.y)/sqrt(pow(b.x-a.x,2)+pow(b.y-a.y, 2));
    double sum = 0;
    unsigned int n = 0;
    for(int i=0;i<(int)sqrt(pow(b.x-a.x, 2)+pow(b.y-a.y, 2));i+=2)
    {
        int y = a.y+i*dy;
        int x = a.x+i*dx;
        if(angle.at<float>(y,x)*(180.0/CV_PI)<361&&angle.at<float>(y,x)*(180.0/CV_PI)>-1)
        {
            double e = acos(abs(dx*cos(angle.at<float>(y,x))+dy*sin(angle.at<float>(y,x))))*(180.0/CV_PI);
            sum +=e;
            n++;
        }
    }
    return sum/(double)n;
}

Point ls::prpoint(Point &center,int d,Point2f &v)
{
    Point pt;
    pt.x = center.x-v.x*d;
    pt.y = center.y-v.y*d;
    return pt;
}

int ls::point_line(Point &p1,Point &p2,Point &tp)
{
    Point ap;
    ap.x = -1;
    float kdx = abs((float)p1.x - (float)p2.x);
    float kdy = abs((float)p1.y - (float)p2.y);
    if(kdy < 20)
        ap.y = p1.y;
    else if(kdx < 20)
    {
        ap.x = p1.x;
        ap.y = -1;
    }
    else
        ap.y = p1.y - (kdy/kdx)*(-1-p1.x);

    float min_size = line_size(ap, p1)<line_size(ap, p2)?line_size(ap, p1):line_size(ap, p2);
    float max_size = line_size(ap, p1)>line_size(ap, p2)?line_size(ap, p1):line_size(ap, p2);
    if(line_size(ap, tp) > max_size)
        return  2;
    else if(line_size(ap, tp)<min_size)
        return  1;
    else
        return  0;
}

int ls::warf(Mat &src,Mat &src1,Mat &angle,LSEG &line1,LSEG &line2,Mat &quad ,int k)
{
    Point2f v1,fv1,vt1,fvt1,v2,fv2,vt2,fvt2;
    Point p1,p2,cp,tp1,tp2,p3,p4,tp3,tp4;
    Point fp1,lp1;
    Point fp2,lp2;
    fp1 = line1[0];
    lp1 = line1[1];
    fp2 = line2[0];
    lp2 = line2[1];
    
    v1.x = ((float)lp1.x - (float)fp1.x)/line_size(fp1,lp1);
    v1.y = ((float)lp1.y - (float)fp1.y)/line_size(fp1,lp1);
    vt1.x = -v1.y;
    vt1.y = v1.x;
    fv1.x = -v1.x;
    fv1.y = -v1.y;
    fvt1.x = -vt1.x;
    fvt1.y = -vt1.y;


    v2.x = ((float)lp2.x - (float)fp2.x)/line_size(fp2,lp2);
    v2.y = ((float)lp2.y - (float)fp2.y)/line_size(fp2,lp2);
    vt2.x = -v2.y;
    vt2.y = v2.x;
    fv2.x = -v2.x;
    fv2.y = -v2.y;
    fvt2.x = -vt2.x;
    fvt2.y = -vt2.y;

    p1 = fp1;
    p2 = lp1;
    p3 = fp2;
    p4 = lp2;

    float dd1 = abs(v2.y*p1.x - v2.x*p1.y + v2.x*p3.y -v2.y*p3.x);
    float dd2 = abs(v2.y*p2.x - v2.x*p2.y + v2.x*p3.y -v2.y*p3.x);
    float dd3 = abs(v1.y*p3.x - v1.x*p3.y + v1.x*p1.y -v1.y*p1.x);
    float dd4 = abs(v1.y*p4.x - v1.x*p4.y + v1.x*p1.y -v1.y*p1.x);

    tp1 = prpoint(p1, dd1, vt2);
    tp2 = prpoint(p2, dd2, vt2);
    tp3 = prpoint(p3, dd3, fvt1);
    tp4 = prpoint(p4, dd4, fvt1);

    int a = point_line(p3, p4, tp1);
    int b = point_line(p3, p4, tp2);
    int c = point_line(p1, p2, tp3);
    int d = point_line(p1, p2, tp4);

    if(a!=0&&b!=0&&a == b)
    {
        return 0;
    }

    if(acos(v1.x*v2.x+v1.y*v2.y)*(180.0/CV_PI)>30)
    {
        return 3;
    }

    Point P1,P2,P3,P4;

    P1 = a?tp3:p1;
    P2 = b?tp4:p2;
    P3 = a?p3:tp1;
    P4 = b?p4:tp2;

    Point p11,p12,p21,p22;
    int d1 = 0,d2 = 0;
    double e_min1=0,e_min2=0;
    double active_d1 = true,active_d2 = true,location_p3 = true,location_p4 = true;
    bool location_p1 = true,location_p2 = true;

    float min_x = 0,max_x = 0,min_y = 0,max_y = 0;
    while((active_d1&&location_p1&&location_p3)||(active_d2&&location_p2&&location_p4))
    {
        Mat image2 = src.clone();

        Point op1,op2,op3,op4;
        p11 = prpoint(P1, d1, v1);
        p12 = prpoint(P2, d2, fv1);
        p21 = prpoint(P3, d1, v2);
        p22 = prpoint(P4, d2, fv2);

        min_x = line_size(p11,p21)<line_size(p12,p22)?line_size(p11,p21):line_size(p12,p22);
        max_x = line_size(p11,p21)>line_size(p12,p22)?line_size(p11,p21):line_size(p12,p22);
        min_y = line_size(p11,p12)<line_size(p21,p22)?line_size(p11,p12):line_size(p21,p22);

        if(min_x < 10||max_x > 150||max_y > 700)
        {
            return 2;
        }

        location_p1 = p11.x<angle.cols&&p11.x>0&&p11.y<angle.rows&&p11.y>0;
        location_p2 = p12.x<angle.cols&&p12.x>0&&p12.y<angle.rows&&p12.y>0;
        location_p3 = p21.x<angle.cols&&p21.x>0&&p21.y<angle.rows&&p21.y>0;
        location_p4 = p22.x<angle.cols&&p22.x>0&&p22.y<angle.rows&&p22.y>0;

        float power1 = power(src, angle, p11, p21);
        float power2 = power(src, angle, p12, p22);

        Point fp1 = prpoint(p11,5,v1);
        Point lp1 = prpoint(p21,5,v1);

        circle(image2, p11, 3, Scalar(255,255,0));
        circle(image2, p21, 3, Scalar(255,255,0));
        circle(image2, fp1, 3, Scalar(255,255,0));
        circle(image2, lp1, 3, Scalar(255,255,0));

        float zpower1 = power(src, angle, p11, fp1)>power(src, angle, p21, lp1)?power(src, angle, p11, fp1):power(src, angle, p21, lp1);

        if(active_d1&&location_p1&&location_p3)
        {
            if(power1 > 60  && zpower1 < 70)
            {
                active_d1 = false;
                e_min1 = abs(power1);
            }
            else
            {
                d1+=1;
            }
        }
        else
        {
//            if(power1>60)
//            {
//                active_d1 = true;
//            }
        }

        Point fp2 = prpoint(p12,5,fv2);
        Point lp2 = prpoint(p22,5,fv2);

        float zpower2 = power(src, angle, p12, fp2)>power(src, angle, p22, lp2)?power(src, angle, p12, fp2):power(src, angle, p22, lp2);
        if(active_d2&&location_p2&&location_p4)
        {
            if(power2 > 60 && zpower2 <70)
            {
                active_d2 = false;
                e_min2 = power2;
            }
            else
            {
                d2+=1;
            }
        }
        else
        {
//           if(power2>60)
//            {
//                active_d2 = true;
//            }
        }
    }

    int x_size = line_size(p11,p12)<line_size(p21,p22)?line_size(p21,p22):line_size(p11,p12);
    int y_size = line_size(p11,p21)<line_size(p12,p22)?line_size(p12,p22):line_size(p11,p21);

    quad = Mat::zeros(x_size*((float)src1.cols/(float)src.cols), y_size*((float)src1.cols/(float)src.cols), CV_8UC3);
    Mat zquad = Mat::zeros(x_size, y_size, CV_8UC3);
    vector<cv::Point2f> corner,quad_pts;
    corner.push_back(Point(p11.x*((float)src1.cols/(float)src.cols),p11.y*((float)src1.cols/(float)src.cols)));
    corner.push_back(Point(p21.x*((float)src1.cols/(float)src.cols),p21.y*((float)src1.cols/(float)src.cols)));
    corner.push_back(Point(p12.x*((float)src1.cols/(float)src.cols),p12.y*((float)src1.cols/(float)src.cols)));
    corner.push_back(Point(p22.x*((float)src1.cols/(float)src.cols),p22.y*((float)src1.cols/(float)src.cols)));

    quad_pts.push_back(cv::Point2f(0,0));
    quad_pts.push_back(cv::Point2f(line_size(p11,p21)*((float)src1.cols/(float)src.cols),0));
    quad_pts.push_back(cv::Point2f(0,line_size(p11,p12)*((float)src1.cols/(float)src.cols)));
    quad_pts.push_back(cv::Point2f(line_size(p12,p22)*((float)src1.cols/(float)src.cols),line_size(p21,p22)*((float)src1.cols/(float)src.cols)));

    Mat transmtx = getPerspectiveTransform(corner, quad_pts);

    warpPerspective(src1, quad, transmtx,quad.size());

    return 1;
}

bool ls::pr_detect(Mat &src,Mat &src1,Mat &angle,vector<LSEG> &oolines,vector<Mat> &result,int k)
{
    if (oolines.size()!=0) {
        QuickSort(oolines, (int)oolines.size(),k);
        int i=0,j=i+1;
        while(i<oolines.size()-1) {
            
            Mat quad;
            
            int bl = warf(src, src1,angle, oolines[i], oolines[j], quad,k);
            
            if(bl == 1)
            {
                result.push_back(quad);
                i=i+1;
                j=i+1;
            }
            else if(bl == 0||bl==3)
            {
                j=j+1;
            }
            else
            {
                i = i + 1;
                j = i + 1;
            }
            if(j == oolines.size())
            {
                i = i+1;
                j = i+1;
            }
        }
        return 1;
    }
    else
        return 0;
}

void ls::findline(Mat &flood,Mat &angle,vector<LSEG> &lines,int k)
{
    vector<vector<Point> > contours;
    findContours(flood, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<LSEG> oolines;
    vector<Point2f> x;
    vector<Point2f> y;
    Mat zflood = Mat::zeros(flood.rows,flood.cols,CV_8U);
    for (int m = 0; m<contours.size(); m++)
    {
        Vec4f ssline;
        vector<cv::Point> zcontours;
        int max_py = 0,min_py = 10000,max_px = 0,min_px = 10000;
        for(int n = 0;n < contours[m].size();n++)
        {
            if(max_py < contours[m][n].y) max_py = contours[m][n].y;
            if(min_py > contours[m][n].y) min_py = contours[m][n].y;
            if(max_px < contours[m][n].x) max_px = contours[m][n].x;
            if(min_px > contours[m][n].x) min_px = contours[m][n].x;
        }

        fitLine(contours[m], ssline, CV_DIST_L2, 0,0.01,0.01);

        Point p0,p1;
        if(ssline[0] == 0)
        {
            p0.x = ssline[2];
            p0.y = min_py;
            p1.x = ssline[2];
            p1.y = max_py;
        }
        else if(ssline[1] == 0)
        {
            p0.x = max_px;
            p0.y = ssline[3];
            p1.x = min_px;
            p1.y = ssline[3];
        }
        else
        {
            p0.x = ((ssline[1]/ssline[0])*ssline[2]+min_py-ssline[3])/(ssline[1]/ssline[0]);
            p0.y = min_py;
            p1.x = ((ssline[1]/ssline[0])*ssline[2]+max_py-ssline[3])/(ssline[1]/ssline[0]);
            p1.y = max_py;
        }

        Point2f v,fv;
        v.x = ((float)p1.x - (float)p0.x)/line_size(p0,p1);
        v.y = ((float)p1.y - (float)p0.y)/line_size(p0,p1);
        fv.x = -v.x;
        fv.y = -v.y;
        Point px,py;
        px = prpoint(p0, 0, v);
        py = prpoint(p1, 0, fv);
        LSEG outline;
        outline.push_back(px);
        outline.push_back(py);
        lines.push_back(outline);
    }
}



void ls::parallelines(vector<LSEG> &lines,vector<LSEG> &outlines)
{
    QuickSort(lines, (int)lines.size(), 4);
    
}

void ls::bookSegmentStart(Mat &src1,vector<Mat> &result)
{
    Mat image;
    
    resize(src1,image,Size(((float)src1.cols/(float)src1.rows)*krows,krows));

    Mat blur_out;

    cvtColor(image, blur_out, CV_RGB2GRAY);

    equalizeHist(blur_out, blur_out);

    medianBlur(blur_out, blur_out, 3);

    GaussianBlur(blur_out, blur_out, Size(5,5),0,0);

    Mat out_x,out_y;

    Sobel(blur_out, out_x, CV_32F, 0, 1 ,3);

    Sobel(blur_out, out_y, CV_32F, 1, 0 ,3);

    blur(out_x, out_x, Size(5,5));

    blur(out_x, out_x, Size(5,5));

    Mat magnitude,angle;

    cartToPolar(out_y, out_x, magnitude, angle);

    vector<double> errors;

    vector<LSEG> lSegs,outlines;

//    cartToPolar(out_y,out_x,magnitude,angle,false);
//    cv::Mat mask = Mat::zeros( blur_out.size(), CV_8UC1 );
//    Ptr<LSDDetector> bd = LSDDetector::createLSDDetector();
//    bd->detect(image, lSegs, 2, 1, mask);

    int R = 3,numMaxLSegs = 1000;

    LSWMS lswms(image.size(), R, numMaxLSegs, false);

    lswms.run(blur_out, lSegs, errors);

    cv::Scalar mean, stddev;

    cv::meanStdDev(errors, mean, stddev);


    Mat floody = Mat::zeros(blur_out.rows, blur_out.cols, CV_8U);
    Mat floodx = Mat::zeros(blur_out.rows, blur_out.cols, CV_8U);
    Mat floodm = Mat::zeros(blur_out.rows, blur_out.cols, CV_8U);
    Mat floodn = Mat::zeros(blur_out.rows, blur_out.cols, CV_8U);
    Mat aa = image.clone();

    vector<LSEG> zlinesx,zlinesy,zlinesm,zlinesn;

    for(int i=0;i<lSegs.size();i++)
    {
//        KeyLine kl = lSegs[i];
//        Point pt1 = Point2f( kl.startPointX, kl.startPointY );
//        Point pt2 = Point2f( kl.endPointX, kl.endPointY );
        Point pt1 = lSegs[i][0];
        Point pt2 = lSegs[i][1];
        if(line_size(pt1,pt2)>100)
        {
            Point2f v,fv;
            v.x = ((float)pt2.x - (float)pt1.x)/line_size(pt1,pt2);
            v.y = ((float)pt2.y - (float)pt1.y)/line_size(pt1,pt2);
            fv.x = -v.x;
            fv.y = -v.y;
            Point p1,p2;
            p1 = prpoint(pt1, 10, fv);
            p2 = prpoint(pt2, 10, v);
            float a = line_jiao(pt1,pt2);
            if(a <= 22.5 || a > 157.5)
            {
                line(floodx, p1, p2, Scalar(255,255,255),3,8);
//                line(aa, p1, p2, Scalar(255,0,0),2,8);
            }
            else if(a > 30  && a <= 80)
            {
                line(floodm, p1, p2, Scalar(255,255,255),3,8);
//                line(aa, p1, p2, Scalar(0,255,0),2,8);
            }
            else if(a >= 112.5 && a < 157.5)
            {
                line(floodn, p1, p2, Scalar(255,255,255),3,8);
//                line(aa, p1, p2, Scalar(0,255,255),2,8);
            }
            else
            {
                line(floody, p1, p2, Scalar(255,255,255),3,8);
//                line(aa, p1, p2, Scalar(0,0,255),2,8);
            }
//            int R = ( rand() % (int) ( 255 + 1 ) );
//            int G = ( rand() % (int) ( 255 + 1 ) );
//            int B = ( rand() % (int) ( 255 + 1 ) );

//            line(aa, pt1,pt2, Scalar(0,0,255),2,8);
        }
    }

    vector< vector<Point> > contours,contoursy,contoursx;

    vector<LSEG> oolinesy,oolinesm,oolinesn,oolinesx;

    findline(floody, angle, oolinesy,0);
    findline(floodm, angle, oolinesm,2);
    findline(floodn, angle, oolinesn,3);
    findline(floodx, angle, oolinesx,1);

    Mat oo_out = image.clone();

    //90度 y,0;
    //0度 x,1;
    //45度 m,2;
    //135度 n,3;

    pr_detect(oo_out,src1,angle,oolinesy,result,0);
    pr_detect(oo_out,src1,angle,oolinesm,result,2);
    pr_detect(oo_out,src1,angle,oolinesn,result,3);
    pr_detect(oo_out,src1,angle,oolinesx,result,1);
}