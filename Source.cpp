#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <stddef.h>
using namespace cv;
using namespace std;
int size = 256;
int grayScale[256] = { 0 };
double probability[256];
Mat image;
ofstream out;
ofstream out1;
ofstream out_triangle1;
ofstream out_triangle2;
int image_numb = 1;
int numb_triangle1 = 0;
int size_points = 0;
int numb_total = 0;

vector <Point2f> tie_points_image1;
vector <Point2f > tie_points_image2;
vector <Point2f> list1;
vector <Point2f> list2;
vector <Point2f> morph_list;
Mat morph_image;
int ** image_matrix1;
int ** image_matrix2;
int ** morph_image_matrix;

//This function is used to write the tie points to a text file
void onMouse(int event, int x, int y, int, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	Point pt = Point(x, y);
	if(image_numb==1)
	out << pt.x <<" "<< pt.y <<"\n";
	else
	out1<< pt.x << " " << pt.y << "\n";

}

//This is used to draw a circle on the tie point
static void draw_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
}

// Draw delaunay triangles
static void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);

	//cout << triangleList.size() << endl;
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		//out_triangle << triangleList[i] << endl;
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		

		

	

		

		// Draw rectangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
				list1.push_back(Point2f(pt[0].x, pt[0].y));
				list1.push_back(Point2f(pt[1].x, pt[1].y));
				list1.push_back(Point2f(pt[2].x, pt[2].y));
				numb_total = numb_total + 3;
				for (int i = 0; i < size_points; i++)
				{
					if (tie_points_image1[i].x == pt[0].x && tie_points_image1[i].y == pt[0].y)
					{
						list2.push_back(tie_points_image2[i]);
						//cout << i << " 0" << endl;
						//out_triangle1 << pt[0].x << " " << pt[0].y << " ";
						out_triangle1 << i << " ";
					}
				}

				for (int i = 0; i < size_points; i++)
				{
					
					if (tie_points_image1[i].x == pt[1].x && tie_points_image1[i].y == pt[1].y)
					{
						list2.push_back(tie_points_image2[i]);
						//cout << i << " 1" << endl;
						//out_triangle1 << pt[1].x << " " << pt[1].y << " ";
						out_triangle1 << i << " ";
					}
				
				}

				for (int i = 0; i < size_points; i++)
				{
					
					if (tie_points_image1[i].x == pt[2].x && tie_points_image1[i].y == pt[2].y)
					{
						list2.push_back(tie_points_image2[i]);
						//cout << i << " 2" << endl;
						//out_triangle1 << pt[2].x << " " << pt[2].y;
						out_triangle1 << i << endl;
					}
				}
				//out_triangle1 << endl;

			    //list1.push_back(Vec6f(pt[0].x, pt[0].y, pt[1].x, pt[1].y, pt[2].x, pt[2].y));
				numb_triangle1++;
				//out_triangle1 << numb_triangle1 << " -- ";
				//out_triangle1 << pt[0].x << " " << pt[0].y <<" "<< pt[1].x << " " << pt[1].y <<" "<< pt[2].x << " " << pt[2].y << endl;
			
			
		}
	}
}




//Reads the tie points from the test file and passes a vector to draw delaunay for making triangles
void Triangle()
{
	out_triangle1.open("Triangle1.txt");
	out_triangle2.open("Triangle2.txt");
	string win_delaunay = "Delaunay Triangulation";
	//string win_voronoi = "Voronoi Diagram";
	string win_delaunay1 = "Delaunay Triangulation";
	//string win_voronoi1 = "Voronoi Diagram";
	// Turn on animation while drawing triangles
	bool animate = true;
	 image_numb = 1;
	// Define colors for drawing.
	Scalar delaunay_color(255, 255, 255), points_color(0, 0, 255);

	// Read in the image.
	Mat img = imread("image1.jpg");

	cout << img.rows << " " << img.cols << endl;
	// Keep a copy around
	Mat img_orig = img.clone();

	// Rectangle to be used with Subdiv2D
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);

	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);

	// Create a vector of points.
	vector<Point2f> points;

	// Read in the points from a text file
	ifstream ifs("File1.txt");
	int x, y;
	while (ifs >> x >> y)
	{
		points.push_back(Point2f(x, y));
	}

	// Insert points into subdiv


	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		subdiv.insert(*it);
		//out_triangle << *it << endl;
		// Show animation
	}

	
	draw_delaunay(img, subdiv, delaunay_color);

	// Draw points
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		draw_point(img, *it, points_color);
	}

	

	// Show results.
	imshow(win_delaunay, img);
	waitKey(0);

}

//This is used to solve 2 linear equations of the form ax+ by = k both
pair<float, float> solve_eq(float a1, float b1, float k1, float a2, float b2, float k2)
{
	//cout << "Solving  " << endl;
	pair<float, float> ans;
	//cout << "Himanshu 3454" << endl;
	ans.first = (k1*b2 - k2*b1) / (b2*a1 - b1*a2);
	//cout << "Himanshu 34544234" << endl;
	//cout << b1*a2 << " " << a2*b1 << endl;
	ans.second = (k1*a2 - k2*a1) / (b1*a2 - a1*b2);
	//cout << "hshfhdsh" << endl;
	return ans;
}

//This is used to calculate area of triangle given 3 points of triangle
float area(int x1, int y1, int x2, int y2, int x3, int y3)
{
	return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0);
}

//this is used to check whether a point (x,y) lies inside the triangle or not
bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y)
{   

	//cout << "AREA" << endl;
   /* Calculate area of triangle ABC */
   float A = area (x1, y1, x2, y2, x3, y3);
 
   /* Calculate area of triangle PBC */  
   float A1 = area (x, y, x2, y2, x3, y3);
 
   /* Calculate area of triangle PAC */  
   float A2 = area (x1, y1, x, y, x3, y3);
 
   /* Calculate area of triangle PAB */   
   float A3 = area (x1, y1, x2, y2, x, y);
   
   /* Check if sum of A1, A2 and A3 is same as A */
   return (A == A1 + A2 + A3);
}

//Used to determine the transformation matrix between the image and the morphed image
void Affine_transformation()
{
	//cout << size_points << endl;
	for (int i = 0; i < numb_total; i = i + 3)
	{
		//cout <<"i= " <<i << endl;
		int a1 = list1[i].x - list1[i + 1].x;
		//cout << "a1= " << a1<<endl;
		int b1 = list1[i].y - list1[i + 1].y;
		//cout << "b1= " << b1<<endl;
		int k1 = morph_list[i].x - morph_list[i + 1].x;
		//cout << "k1= " << k1 << endl;
		int a2 = list1[i].x - list1[i + 2].x;
		//cout << "a2= " << a2<<endl;
		int b2 = list1[i].y - list1[i + 2].y;
		//cout << "b2= " << b2<<endl;
		int k2 = morph_list[i].x - morph_list[i + 2].x;
		//cout << morph_list[i].first << " " << morph_list[i + 2].first << endl;
		//cout << "k2= " << k2<< endl;
		pair <float, float> row1 = solve_eq(a1,b1 , k1, a2, b2, k2);
		pair <float, float> row2 = solve_eq(a1, b1, morph_list[i].y - morph_list[i + 1].y, a2, b2, morph_list[i].y - morph_list[i + 2].y);
		int m = morph_list[i].x - row1.first*list1[i].x - row1.second*list1[i].y;
		int n = morph_list[i].y - row2.first*list1[i].x - row2.second*list1[i].y;
		float t[2][2] = { { row1.first, row1.second },
		{ row2.first,row2.second} };
		//cout << "HIIIIII" << endl;
		float s[2][1] = { {m},{n} };
		Mat T = Mat(2, 2, CV_32FC1, t);
		Mat S = Mat(2, 1, CV_32FC1, s);
		float res[2][1] = { {0.0},{0.0} };
		Mat RES = Mat(2, 1, CV_32FC1, res);
		for (int j = 0; j <image.rows; j++)
		{
			//cout << "HIMANSHU TOLANI " << endl;
			for (int k = 0; k < image.cols; k++)
			{
				//cout << "HELLO" << endl;
				float x[2][1] = { { j },{ k } };
				Mat X = Mat(2, 1, CV_32FC1, x);
				
				RES = T*X + S;
				//cout << "jdfsd" << endl;
				cout << T << endl;
				cout << "----------------" << endl;
				cout << S << endl;
				cout << "__________________________"<<endl;
				cout << RES << endl;
				cout << "///////////////////////////" << endl;
				//int c1 = 0; int c2 = 0;
				//cout << (int)(RES.at<int>(0, 0)) << "/////////" << endl;
				int c1 =abs((int)(RES.at<int>(0, 0)));
				int c2 = abs((int)(RES.at<int>(1, 0)));
				//cout << "c1= " << c1 << " c2= " << c2 << endl;
				if (isInside(list1[i].x, list1[i].y, list1[i + 1].x, list1[i + 1].y, list1[i + 2].x, list1[i + 2].y, j, k))
				{
					//cout << "WELCOME" << endl;
					
					if (isInside(morph_list[i].x, morph_list[i].y, morph_list[i + 1].x, morph_list[i + 1].y, morph_list[i + 2].x, morph_list[i + 2].y, c1, c2))
					{
						morph_image_matrix [c1][c2] = 0.5*image_matrix1[j][k]+0.8*image_matrix2[j][k];
						morph_image.at<uchar>(c1, c2) = (uchar)(morph_image_matrix[c1][c2]);
					}
				}
			}
		}

		//cout << T << endl << S;
		
	}

	namedWindow("DisplayHALF", WINDOW_AUTOSIZE);
	imshow("DisplayHALF", morph_image);
	waitKey(0);

	//cout << "hfhfhfh " << i;
}

//This is used to find the minimum area rectangle which encloses the given triangle
Rect bounding_Rectangle(vector<Point2f> &t) {

	int leftmost = t[0].x;
	int topmost = t[0].y;
	int rightmost = t[0].x;
	int bottommost = t[0].y;

	for (int i = 0; i< 3; i++) {

		if (t[i].x < leftmost) {
			leftmost = t[i].x;
		}

		if (t[i].y < topmost) {
			topmost = t[i].y;
		}


		if (t[i].x > rightmost) {
			rightmost = t[i].x;
		}

		if (t[i].y > bottommost) {
			bottommost = t[i].y;
		}

	}


	return Rect(leftmost, topmost, rightmost + 1 - leftmost, bottommost + 1 - topmost);

}

//---------------------------------------------------------------------------------------------


// Apply affine transform calculated using image1Tri and MorphedTri to src
void apply_Transform(Mat &morphedImage, Mat &image1, vector<Point2f> &image1Tri, vector<Point2f> &MorphedTri)
{
	Mat warped = getAffineTransform(image1Tri, MorphedTri);
	//std::cout << image1Tri[0].x << " " << image1Tri[0].y << std::endl;
	warpAffine( image1, morphedImage, warped, morphedImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Applies affine transform to the triangle and alpha blends the image
void morph_Tri(Mat &img1, Mat &img2, Mat &img, vector<Point2f> &t1, vector<Point2f> &t2, vector<Point2f> &t, double alpha)
{

	// Find bounding rectangle for each triangle
	Rect r = bounding_Rectangle(t);
	Rect r1 = bounding_Rectangle(t1);
	Rect r2 = bounding_Rectangle(t2);

	// Offset points by left top corner of the respective rectangles
	vector<Point2f> t1Rect, t2Rect, tRect;
	vector<Point> tRectInt;
	for (int i = 0; i < 3; i++)
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y)); // for fillConvexPoly

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	// Get mask by filling triangle
	Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
	fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);
	img2(r2).copyTo(img2Rect);

	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

	apply_Transform(warpImage1, img1Rect, t1Rect, tRect);
	apply_Transform(warpImage2, img2Rect, t2Rect, tRect);

	// Alpha blend rectangular patches
	Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

	// Copy triangular region of the rectangular patch to the output image
	multiply(imgRect, mask, imgRect);
	multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
	img(r) = img(r) + imgRect;


}

/*********** This function is used to mark the tie points ***************/
void Mark_tie_points(string s1,string s2)
{
	string s = s1;
	out.open("File1.txt");
	out1.open("File2.txt");
	image = imread(s, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return ;
	}



	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Display window", onMouse, 0);
	imshow("Display window", image);                   // Show our image inside it.
	waitKey(0);                                          // Wait for a keystroke in the window
	image_numb = 2;
	s = s2;
	image = imread(s, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return ;
	}



	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Display window", onMouse, 0);
	imshow("Display window", image);                   // Show our image inside it.
	waitKey(0);
}

//Calculates the average tie points and calls morph_tri on every triangle
void morph_DO(string s1, string s2)
{
	double alpha = 0.5;
	int q = 0;
	for (alpha = 0.05; alpha <= 1.0; alpha = alpha + 0.05)
	{
		q++;
		//Read input images
		Mat img1 = imread(s1);
		Mat img2 = imread(s2);

		//convert Mat to float data type
		img1.convertTo(img1, CV_32F);
		img2.convertTo(img2, CV_32F);


		//empty average image
		Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);


		//Read points
		//vector<Point2f> points1 = readPoints("File1.txt");
		//vector<Point2f> points2 = readPoints("File2.txt");
		vector<Point2f> points;

		//compute weighted average point coordinates
		for (int i = 0; i < tie_points_image1.size(); i++)
		{
			float x, y;
			x = (1 - alpha) * tie_points_image1[i].x + alpha * tie_points_image2[i].x;
			y = (1 - alpha) * tie_points_image1[i].y + alpha * tie_points_image2[i].y;

			points.push_back(Point2f(x, y));

		}


		//Read triangle indices
		ifstream ifs("Triangle1.txt");
		int m, n, o;

		while (ifs >> m >> n >> o)
		{
			// Triangles
			vector<Point2f> t1, t2, t;

			// Triangle corners for image 1.
			t1.push_back(tie_points_image1[m]);
			t1.push_back(tie_points_image1[n]);
			t1.push_back(tie_points_image1[o]);

			// Triangle corners for image 2.
			t2.push_back(tie_points_image2[m]);
			t2.push_back(tie_points_image2[n]);
			t2.push_back(tie_points_image2[o]);

			// Triangle corners for morphed image.
			t.push_back(points[m]);
			t.push_back(points[n]);
			t.push_back(points[o]);

			morph_Tri(img1, img2, imgMorph, t1, t2, t, alpha);

		}

		// Display Result
		imshow("Morphed Face", imgMorph / 255.0);
		string str = "Morph_image12" + to_string(q) + ".png";
		imwrite(str, imgMorph);
		waitKey(500);
	}
}
int main()
{


	
	string s1 = "image1.jpg";
	string s2 = "image2.jpg";
	// Mark_tie_points(s1,s2);
	/*image = imread(s1, CV_LOAD_IMAGE_GRAYSCALE);
	morph_image = image.clone();
	image_matrix1 = new int*[image.rows];
	for (int i = 0; i<image.rows; i++)
		image_matrix1[i] = new int[image.cols];

	image_matrix2 = new int*[image.rows];
	for (int i = 0; i<image.rows; i++)
		image_matrix2[i] = new int[image.cols];

	morph_image_matrix = new int*[image.rows];
	for (int i = 0; i<image.rows; i++)
		morph_image_matrix[i] = new int[image.cols];

	for (int i = 0; i <image.rows; i++)
	{
		for (int j = 0; j <image.cols; j++)
		{
			image_matrix1[i][j] = (int)(image.at<uchar>(i, j));
			//cout << image_matrix[i][j] << " ";
		}
		//cout << endl;
	}
	image= imread("image2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i <image.rows; i++)
	{
		for (int j = 0; j <image.cols; j++)
		{
			image_matrix2[i][j] = (int)(image.at<uchar>(i, j));
			//cout << image_matrix[i][j] << " ";
		}
		//cout << endl;
	}*/

	//cout << "Tolni" << endl;
	ifstream ifs1("File1.txt");
	int x, y;
	
	while (ifs1 >> x >> y)
	{
		tie_points_image1.push_back(Point2f(x, y));
		size_points++;
	}
	//cout << "size-- " << size_points << endl;

	ifstream ifs2("File2.txt");
	while (ifs2 >> x >> y)
	{
		tie_points_image2.push_back(Point2f(x, y));
	}
	//cout << "HIMANSHU" << endl;
	Triangle();
	//cout << "---------MORPH IMAGE-----------------" << endl;
	//cout << "TOLANI" << endl;
	int a, b;
   	for (int i = 0; i <numb_total; i++)
	{
		
		a = (list1[i].x + list2[i].x) / 2;
		b = (list1[i].y + list2[i].y) / 2;
		morph_list.push_back(Point2f(a, b));
	}
	cout << numb_triangle1 << " hhh "  << endl;
	cout << list1.size() << endl;

	morph_DO(s1, s2);
	//pair <int, int> answer = solve_eq(2, 3, 12, 3, 2, 18);
	//cout << answer.first<<" "<< answer.second;
	//Affine_transformation();

	//_---------------------------------------------------------------------------------------//
	

	//namedWindow("Display window", WINDOW_AUTOSIZE);
	//imshow("Display window", image);
	//waitKey(0);

	
	// Wait for a keystroke in the window

	
	return 0;



}