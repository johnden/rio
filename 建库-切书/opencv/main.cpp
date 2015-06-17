//-----------------------------------【头文件包含部分】---------------------------------------
//	     描述：包含程序所依赖的头文件
//----------------------------------------------------------------------------------------------
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
/////////////////////////////////////////////////////////////////////////////////////////////////
float a = 20,b = 40,c = 30;
float krows = 640.0;

int ca(LSEG &line,int k)
{
    float kdx = (float)line[0].x - (float)line[1].x;
    float kdy = (float)line[0].y - (float)line[1].y;
    if(line[0].x == line[1].x)
    {
        if(k==0)
            return line[0].x;
        else if(k==1)
            return line[0].x;
        else
            return line[0].x;
    }
    else
    {
        if(k==0)
            return (line[0].x+line[1].x)*0.5;
        else if(k==1)
            return ((kdy/kdx)*line[0].x-line[0].y)/(kdy/kdx);
        else
            return ((kdy/kdx)*line[0].x+krows-line[0].y)/(kdy/kdx);
    }
}

void run(vector<LSEG> &lines1,int left,int right,int k)
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

void QuickSort(vector<LSEG> &lines1,int Count,int k)
{
    run(lines1, 0,Count-1,k);
}

double line_size(LSEG line)
{
    return sqrt(pow(line[0].x - line[1].x,2)+pow(line[0].y - line[1].y,2));
}

Point prpoint(Point &center,int d,Point2f &v)
{
    Point pt;
    pt.x = center.x-v.x*d;
    pt.y = center.y-v.y*d;
    return pt;
}

double power(Mat &src,Mat &angle,Point &a,Point &b)
{
    
    double dx = (b.x-a.x)/sqrt(pow(b.x-a.x,2)+pow(b.y-a.y, 2));
    double dy = (b.y-a.y)/sqrt(pow(b.x-a.x,2)+pow(b.y-a.y, 2));
    double sum = 0;
    unsigned int n = 0;
    Mat out = src.clone();
    for(int i=0;i<(int)sqrt(pow(b.x-a.x, 2)+pow(b.y-a.y, 2));i+=10)
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

int warf(Mat src1,Mat &src,Mat &angle,LSEG &line1,LSEG &line2,Mat &quad)
{
    Point2f v1,fv1,vt1,fvt1,v2,fv2,vt2,fvt2;
    Point p1,p2,zp2;
    
    v1.x = ((float)line1[1].x - (float)line1[0].x)/line_size(line1);
    v1.y = ((float)line1[1].y - (float)line1[0].y)/line_size(line1);
    vt1.x = -v1.y;
    vt1.y = v1.x;
    fv1.x = -v1.x;
    fv1.y = -v1.y;
    fvt1.x = -vt1.x;
    fvt1.y = -vt1.y;
    v2.x = ((float)line2[1].x - (float)line2[0].x)/line_size(line2);
    v2.y = ((float)line2[1].y - (float)line2[0].y)/line_size(line2);
    vt2.x = -v2.y;
    vt2.y = v2.x;
    fv2.x = -v2.x;
    fv2.y = -v2.y;
    fvt2.x = -vt2.x;
    fvt2.y = -vt2.y;
    p1.x = (line1[0].x + line1[1].x)/2.0;
    p1.y = (line1[0].y + line1[1].y)/2.0;
    zp2.x = (line2[0].x + line2[1].x)/2.0;
    zp2.y = (line2[0].y + line2[1].y)/2.0;
    float dd = abs(v2.y*p1.x - v2.x*p1.y + v2.x*zp2.y -v2.y*zp2.x);
    
    p2 = prpoint(p1, dd, vt1);
    
    Point fp1 = prpoint(zp2, dd, fvt1);
    
    Point p11,p12,p21,p22;
    int d1 = 70,d2 = 70;
    double e_min1=0,e_min2=0;
    double active_d1 = true,active_d2 = true,location_p3 = true,location_p4 = true;
    bool location_p1 = true,location_p2 = true;
    
    float min_x = 0,max_x = 0,min_y = 0,max_y = 0;
    while((active_d1&&location_p1&&location_p3)||(active_d2&&location_p2&&location_p4))
    {
        Mat image2 = src.clone();
        
        p11 = prpoint(p1, d1, v1);
        p12 = prpoint(p1, d2, fv1);
        p21 = prpoint(p2, d1, v2);
        p22 = prpoint(p2, d2, fv2);
        
        LSEG fline,lline,xline,yline;
        fline.push_back(p11);
        fline.push_back(p21);
        lline.push_back(p12);
        lline.push_back(p22);
        xline.push_back(p11);
        xline.push_back(p12);
        yline.push_back(p21);
        yline.push_back(p22);
        
        min_x = line_size(fline)<line_size(lline)?line_size(fline):line_size(lline);
        max_x = line_size(fline)>line_size(lline)?line_size(fline):line_size(lline);
        min_y = line_size(xline)<line_size(yline)?line_size(xline):line_size(yline);
        max_y = line_size(xline)>line_size(yline)?line_size(xline):line_size(yline);
        if(min_x < 5||max_x > 200||max_y > 700||min_y<20)
        {
            return 2;
        }
        
        
        location_p1 = p11.x<angle.cols&&p11.x>0&&p11.y<angle.rows&&p11.y>0;
        location_p2 = p12.x<angle.cols&&p12.x>0&&p12.y<angle.rows&&p12.y>0;
        location_p3 = p21.x<angle.cols&&p21.x>0&&p21.y<angle.rows&&p21.y>0;
        location_p4 = p22.x<angle.cols&&p22.x>0&&p22.y<angle.rows&&p22.y>0;
        
        float power1 = power(src, angle, p11, p21);
        float power2 = power(src, angle, p12, p22);
        
        if(active_d1&&location_p1&&location_p3)
        {
            Point fp1 = prpoint(p11,5,v1);
            Point lp1 = prpoint(p21,5,v1);
            
            float zpower = power(src, angle, p11, fp1)>power(src, angle, p21, lp1)?power(src, angle, p11, fp1):power(src, angle, p21, lp1);
            
            if(abs(power1) > a  && abs(zpower) < b)
            {
                active_d1 = false;
                e_min1 = abs(power1);
            }
            else
            {
                d1+=2;
            }
        }
        else
        {
            if(abs(power1)<c)
            {
                active_d1 = true;
            }
        }
        
        if(active_d2&&location_p2&&location_p4)
        {
            Point fp2 = prpoint(p12,5,fv2);
            Point lp2 = prpoint(p22,5,fv2);
            
            float zpower = power(src, angle, p12, fp2)>power(src, angle, p22, lp2)?power(src, angle, p12, fp2):power(src, angle, p22, lp2);
            
            if(abs(power2) > a&& abs(zpower)<b)
            {
                active_d2 = false;
                e_min2 = abs(power2);
            }
            else
            {
                d2+=2;
            }
        }
        else
        {
            if(abs(power2)<c)
            {
                active_d2 = true;
            }
        }
    }

    LSEG q_line1,q_line2,q_line3,q_line4;
    q_line1.push_back(p11);
    q_line1.push_back(p12);
    q_line2.push_back(p21);
    q_line2.push_back(p22);
    q_line3.push_back(p11);
    q_line3.push_back(p21);
    q_line4.push_back(p12);
    q_line4.push_back(p22);
    
    int x_size = line_size(q_line1)<line_size(q_line2)?line_size(q_line2):line_size(q_line1);
    int y_size = line_size(q_line3)<line_size(q_line4)?line_size(q_line4):line_size(q_line3);

        quad = Mat::zeros(x_size*((float)src1.cols/(float)src.cols), y_size*((float)src1.cols/(float)src.cols), CV_8UC3);
        
        Mat zquad = Mat::zeros(x_size, y_size, CV_8UC3);
        vector<cv::Point2f> corner,quad_pts;
        corner.push_back(Point(p11.x*((float)src1.cols/(float)src.cols),p11.y*((float)src1.cols/(float)src.cols)));
        corner.push_back(Point(p21.x*((float)src1.cols/(float)src.cols),p21.y*((float)src1.cols/(float)src.cols)));
        corner.push_back(Point(p12.x*((float)src1.cols/(float)src.cols),p12.y*((float)src1.cols/(float)src.cols)));
        corner.push_back(Point(p22.x*((float)src1.cols/(float)src.cols),p22.y*((float)src1.cols/(float)src.cols)));

        quad_pts.push_back(cv::Point2f(0,0));
        quad_pts.push_back(cv::Point2f(line_size(q_line3)*((float)src1.cols/(float)src.cols),0));
        quad_pts.push_back(cv::Point2f(0,line_size(q_line1)*((float)src1.cols/(float)src.cols)));
        quad_pts.push_back(cv::Point2f(line_size(q_line4)*((float)src1.cols/(float)src.cols),line_size(q_line2)*((float)src1.cols/(float)src.cols)));

        Mat transmtx = getPerspectiveTransform(corner, quad_pts);
        
        warpPerspective(src1, quad, transmtx,quad.size());
        
        return 1;
}

bool pr_detect(Mat src1,Mat &src,Mat &angle,vector<LSEG> &oolines,vector<Mat> &result,int k)
{
    if (oolines.size()!=0) {
        QuickSort(oolines, (int)oolines.size(),k);
        int i=0,j=i+1;
        while(i<oolines.size()-1) {
            
            Mat quad;
            
            int bl = warf(src1,src, angle, oolines[i], oolines[j], quad);
            
            if(bl == 1)
            {
                result.push_back(quad);
                i=i+1;
                j=i+1;
            }
            else if(bl == 0)
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

void findline(Mat &flood,vector<LSEG> &lines)
{
    vector<vector<Point> > contours;
    findContours(flood, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    vector<LSEG> oolines;
    vector<Point2f> x;
    vector<Point2f> y;
    for (int m = 0; m<contours.size(); m++)
    {
        Vec4f ssline;
        fitLine(contours[m], ssline, CV_DIST_L2, 0,0.01,0.01);
        int max_py = 0,min_py = 10000,max_px = 0,min_px = 10000;
        for(int n = 0;n < contours[m].size();n++)
        {
            if(max_py < contours[m][n].y) max_py = contours[m][n].y;
            if(min_py > contours[m][n].y) min_py = contours[m][n].y;
            if(max_px < contours[m][n].x) max_px = contours[m][n].x;
            if(min_px > contours[m][n].x) min_px = contours[m][n].x;
        }
        int x0,y0,x1,y1;
        if(ssline[0]!=0&&ssline[1]!=0&&(ssline[1]>0.5||ssline[1]<-0.5))
        {
            x0 = ((ssline[1]/ssline[0])*ssline[2]+min_py-ssline[3])/(ssline[1]/ssline[0]);
            y0 = min_py;
            x1 = ((ssline[1]/ssline[0])*ssline[2]+max_py-ssline[3])/(ssline[1]/ssline[0]);
            y1 = max_py;
        }
        else if(ssline[0]!=0&&ssline[1]!=0&&ssline[0]>0&&(ssline[1]<0.5&&ssline[1]>-0.5))
        {
            x0 = max_px;
            y0 = ssline[3]+(ssline[1]/ssline[0])*(max_px - ssline[2]);
            x1 = min_px;
            y1 = ssline[3]+(ssline[1]/ssline[0])*(min_px - ssline[2]);
        }
        else if(ssline[0] == 0)
        {
            x0 = ssline[2];
            y0 = min_py;
            x1 = ssline[2];
            y1 = max_py;
        }
        else
        {
            x0 = max_px;
            y0 = ssline[3];
            x1 = min_px;
            y1 = ssline[3];
        }
        
        LSEG outline;
        outline.push_back(Point(x0,y0));
        outline.push_back(Point(x1,y1));
        if(line_size(outline)>krows*0.2&&line_size(outline)<krows)
            lines.push_back(outline);
    }
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////主函数//////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
int main()
{
    Mat image;
    //载入原始图
    Mat src1 = imread("/Users/incer/Documents/imageproject/textimages/gongsi/12.jpg");
    //////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////改图像路径///////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////
    resize(src1,image,Size(((float)src1.cols/(float)src1.rows)*krows,krows));
    
    Mat out = image.clone();
    
    Mat out1 = image.clone();
    
    Mat blur_out;

    cvtColor(image, blur_out, CV_RGB2GRAY);
    
//    blur(blur_out, blur_out, Size(5,5));
    
    equalizeHist(blur_out, blur_out);
    int R = 1,numMaxLSegs = 10000;
    
    LSWMS lswms(image.size(), R, numMaxLSegs, false);
    
    vector<double> errors;
    
    vector<LSEG> lSegs,outlines;
    
    lswms.run(blur_out, lSegs, errors);
    
    cv::Scalar mean, stddev;
    
    cv::meanStdDev(errors, mean, stddev);
    
    Mat floody = blur_out.clone();
    
    floody.setTo(0);
    
    for(int i=0;i<lSegs.size();i++)
    {
        if(abs(lSegs[i][0].x-lSegs[i][1].x)<10&&abs(lSegs[i][0].y-lSegs[i][1].y)>50)
        {
            Point2f v,fv;
            v.x = ((float)lSegs[i][1].x - (float)lSegs[i][0].x)/line_size(lSegs[i]);
            v.y = ((float)lSegs[i][1].y - (float)lSegs[i][0].y)/line_size(lSegs[i]);
            fv.x = -v.x;
            fv.y = -v.y;
            Point p1,p2;
            p1 = prpoint(lSegs[i][0], 10, v);
            p2 = prpoint(lSegs[i][1], 10, fv);
            line(floody, p1, p2, Scalar(255,255,255),3,8);
        }
    }
    
    imshow("fsdfs", floody);
    vector< vector<Point> > contours,contoursy,contoursx;
   
    vector<LSEG> oolines;
    
    findline(floody, oolines);
    
    for(int i=0;i<oolines.size();i++)
    {
//        if(abs(oolines[i][0].y-oolines[i][1].y)>abs(oolines[i][0].x-oolines[i][1].x)&&line_size(oolines[i])>100)
        {
            line(image, oolines[i][0], oolines[i][1], Scalar(0,0,255),2,8);
        }
    }
    
    imshow("fsdfsds", image);
    
    Mat out_x,out_y;
    
    Sobel(blur_out, out_x, CV_32F, 0, 1 ,3);
    
    Sobel(blur_out, out_y, CV_32F,1, 0 ,3);
    
    Mat magnitude,angle;
    
    cartToPolar(out_y,out_x,magnitude,angle,false);
    
    Mat oo_out = out.clone();
    
    vector<Mat> result;
    
    pr_detect(src1,oo_out, angle, oolines, result, 0);

    for(int i = 0;i<result.size();i++)
    {
        char t[256];
        string s;
        sprintf(t, "%d", i+0);
        s = t;
        imshow("fsdfsdf", result[i]);
        waitKey(10);
        imwrite("/Users/incer/Documents/imageproject/textimages/out/"+s+".jpg", result[i]);
    }
    
//    waitKey(0);
    
    return 0;
}