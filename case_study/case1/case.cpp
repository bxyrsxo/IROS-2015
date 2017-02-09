#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void white_noise( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void partial_cloud( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
void Cloud_Transform( pcl::PointCloud<pcl::PointXYZ>::Ptr& in, pcl::PointCloud<pcl::PointXYZ>::Ptr& out, double a, double b, double c);
void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff);
void EigenVector_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff);
void indirect_scheme ( Eigen::VectorXd& th, double& u, double& v, double& w);
void indirect_scheme1( Eigen::VectorXd& th, double& u, double& v, double& w);
void Direction_Decision ( double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
void Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi);
double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3);

pcl::ModelCoefficients::Ptr tempvar (new pcl::ModelCoefficients);

char fn_theta[] = "theta.txt";
char fn_thdot[] = "thdot.txt";
char fn_uvw  [] = "uvw.txt";
char fn_uvw1 [] = "uvw1.txt";

fstream fp_theta, fp_thdot, fp_uvw, fp_uvw1;

int
main (int argc, char** argv)
{

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
	viewer->addCoordinateSystem (1.0);

	// generate the point cloud--------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	Data_Generation( cloud);

	// partial cloud
//	partial_cloud( cloud);

	// add white noise
//	white_noise( cloud);

	// cloud transform
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ModelCoefficients::Ptr adaptive_coeff (new pcl::ModelCoefficients);

	double a = 60;
	double b = -120;
	double c = 150;

	fp_theta.open(fn_theta, ios::out);
	fp_thdot.open(fn_thdot, ios::out);
	fp_uvw.open(fn_uvw, ios::out);
	fp_uvw1.open(fn_uvw1, ios::out);


	Cloud_Transform( cloud, cloud_trans, pcl::deg2rad(a), pcl::deg2rad(b), pcl::deg2rad(c)); 
	// adaptive estimation
	// Adaptive_Estimation( cloud_trans, adaptive_coeff);
	EigenVector_Estimation( cloud_trans, adaptive_coeff);

    // put the cylinder model on the visualizer-------------------------------------
/*
viewer->addCylinder(*adaptive_coeff, "cylinder1");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "cylinder1");
	viewer->addCylinder(*tempvar, "cylinder");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "cylinder");
	
	viewer->addPointCloud(cloud_trans);
*/	
	// display---------------------------------------------------------------------
	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
	fp_theta.close();
	fp_thdot.close();
	fp_uvw.close();
	fp_uvw1.close();


	return 0;
}

void Data_Generation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	pcl::ScopeTime t ("Data Generation");
	for (float z(0.0); z <= 5.0; z += 0.25)
		for (float angle(0.0); angle <= 360.0; angle += 5.0)
		{
			pcl::PointXYZ point;
			point.x = 1.0*cosf (pcl::deg2rad(angle));
			point.y = 1.0*sinf (pcl::deg2rad(angle));
			point.z = z;
			pc->points.push_back(point);
		}
}

void Adaptive_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff)
{
	pcl::ScopeTime t ("Adaptive Estimation");
	const double radius = 1.0;
	const double L = 5.0;

	Eigen::VectorXd th(6);
	th << 1, 0, 1, 0, 0, 0;
	double u = 0, v = 0, w = 1;
	double u1, v1, w1;
	int cloud_size = pc->size();

	pcl::PointXYZ J0(0.0,0.0,0.0);
	double sampling_time = 0.001;
	double gamma = 5;
	srand((unsigned)time(NULL));
	

    pcl::PointXYZ Pi;
	int k;
	for( k = 0; k < 20000; k++)
	{
		Eigen::VectorXd phi(6);
		do{
			double random = (double)(rand()) / (RAND_MAX + 1.0);
			unsigned int index = static_cast<unsigned int>(random*cloud_size);
			Pi = pc->points[index];
		}while( Pi.x*Pi.x + Pi.y*Pi.y + Pi.z*Pi.z < 2);
		
		double a, b, c;
		a = Pi.x - J0.x;
		b = Pi.y - J0.y;
		c = Pi.z - J0.z;
	
		do{
			double random = (double)(rand()) / (RAND_MAX + 1.0);
			unsigned int index = static_cast<unsigned int>(random*cloud_size);
			Pi = pc->points[index];
		}while( Pi.x*Pi.x + Pi.y*Pi.y + Pi.z*Pi.z < 2);

		double a1, b1, c1;
		a1 = Pi.x - J0.x;
		b1 = Pi.y - J0.y;
		c1 = Pi.z - J0.z;
		
		phi(0) = a*a - a1*a1;
		phi(1) = b*b - b1*b1;
		phi(2) = c*c - c1*c1;
		phi(3) = a*b - a1*b1;
		phi(4) = b*c - b1*c1;
		phi(5) = a*c - a1*c1;
		
		double z_head = th.transpose()*phi;
		double error = -z_head;
		
		Eigen::VectorXd th_dot(6);
		th_dot = gamma*phi*error;
		// adaptive law with projection
		for( int i = 0; i < 3; i++)
			if( th(i) < 0 && th_dot(i) < 0)
				th_dot(i) = 0;
				
		// estimated theta update
		th = th + th_dot*sampling_time;
		// set zero if theta smaller than zero
		for( int i = 0; i < 3; i++)
			if( th(i) < 0)
				th(i) = 0;


		indirect_scheme( th, u, v, w);
		Direction_Decision( u, v, w, J0, Pi);
	
		indirect_scheme1(th, u1, v1, w1);
		Direction_Decision1( th, u1, v1, w1, J0, Pi);
	
		// output file
		fp_theta<<th.transpose()<<endl;
		fp_thdot<<th_dot.transpose()<<endl;
		fp_uvw<<u<<" "<<v<<" "<<w<<endl;
		fp_uvw1<<u1<<" "<<v1<<" "<<w1<<endl;

	}

	cout<<th.transpose()<<endl;


	double pose_x, pose_y, pose_z;
	double pose_x1, pose_y1, pose_z1;
	pose_x = u*L + J0.x;
	pose_y = v*L + J0.y;
	pose_z = w*L + J0.z;

	pose_x1 = u1*L + J0.x;
	pose_y1 = v1*L + J0.y;
	pose_z1 = w1*L + J0.z;
	
	coeff->values.push_back(0);
	coeff->values.push_back(0);
	coeff->values.push_back(0);
	coeff->values.push_back(pose_x);
	coeff->values.push_back(pose_y);
	coeff->values.push_back(pose_z);
	coeff->values.push_back(radius);
	
	tempvar->values.push_back(0);
	tempvar->values.push_back(0);
	tempvar->values.push_back(0);
	tempvar->values.push_back(pose_x1);
	tempvar->values.push_back(pose_y1);
	tempvar->values.push_back(pose_z1);
	tempvar->values.push_back(radius);
}


void indirect_scheme( Eigen::VectorXd& th, double& u, double& v, double& w)
{
	double beta, alpha;
	beta = atan2(th(4), th(5));
	alpha = atan2(th(3), th(5)*sin(beta));

	u = sin(alpha)*cos(beta);
	v = sin(alpha)*sin(beta);
	w = cos(alpha);
}

void Cloud_Transform( pcl::PointCloud<pcl::PointXYZ>::Ptr& in, pcl::PointCloud<pcl::PointXYZ>::Ptr& out, double a, double b, double c)
{
	// input data transformation
	Eigen::Matrix3d Rz, Ry, Rx, R;
	Rz << cos(a), -sin(a), 0, sin(a),  cos(a), 0, 0, 0, 1;
	Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
	Rx << 1, 0, 0, 0, cos(c), -sin(c), 0, sin(c), cos(c); 
	R = Rz*Ry*Rx;

	Eigen::Vector3d pi_vec, p_vec;
	for( int i = 0; i < in->size(); i++)
	{
		pi_vec(0) = in->points[i].x;
		pi_vec(1) = in->points[i].y;
		pi_vec(2) = in->points[i].z;
		
		p_vec = R*pi_vec;
		out->push_back( pcl::PointXYZ( p_vec(0), p_vec(1), p_vec(2)));
	}
}

double angle_bt_vectors( pcl::PointXYZ& p1, pcl::PointXYZ& p2, pcl::PointXYZ& p3)
{
	Eigen::Vector3d v1(p2.x-p1.x, p2.y-p1.y, p2.z-p1.z);
	Eigen::Vector3d v2(p3.x-p1.x, p3.y-p1.y, p3.z-p1.z);
	double r = v1.dot(v2);
	r = r/v1.norm()/v2.norm();
	return acos(r);
}

void Direction_Decision( double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi)
{
	pcl::PointXYZ J1, J2;
	J1 = pcl::PointXYZ(u,v,w);
	J2 = pcl::PointXYZ(-u,-v,-w);
	
	double a1, a2;
	a1 = angle_bt_vectors( J0, J1, Pi);		
	a2 = angle_bt_vectors( J0, J2, Pi);		

	if( a1 > a2)
	{
		u = -u;	v = -v; w = -w;
	}
}

void white_noise( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	rng.seed(time(NULL));
	
	// simulate rolling a die
	boost::normal_distribution<> nd(0.0, 0.1);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
	
	for( int i = 0; i < pc->size(); i++)
	{
		pc->points[i].x +=  var_nor();	
		pc->points[i].y +=  var_nor();	
		pc->points[i].z +=  var_nor();	
	}
	
}

void partial_cloud( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc)
{
	srand( (unsigned) time(NULL));	
	int cloud_size = pc->size();
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cp (new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::copyPointCloud( *pc, *cp);
	pc->clear();

	int num_points = cloud_size / 8;
	
	for( int i = 0; i < num_points; i++)
	{	
		double random = (double)(rand()) / (RAND_MAX + 1.0);
		unsigned int index = static_cast<unsigned int>(random*cloud_size);
		pc->push_back( cp->points[index]);
	}
}

void indirect_scheme1( Eigen::VectorXd& th, double& u, double& v, double& w)
{	
	if( th(0) > 1)
		th(0) = 1;
	if( th(1) > 1)
		th(1) = 1;
	if( th(2) > 1)
		th(2) = 1;

	double s = 2.0/(th(0)+th(1)+th(2));
	double t0 = th(0)*s;
	double t1 = th(1)*s;
	double t2 = th(2)*s;

	u = sqrt( 1 - t0);
	v = sqrt( 1 - t1);
	w = sqrt( 1 - t2);
}

void Direction_Decision1( Eigen::VectorXd& th, double& u, double& v, double& w, pcl::PointXYZ& J0, pcl::PointXYZ& Pi)
{
	pcl::PointXYZ J1, J2;
	// determine the direction
	if( fabs( th(3)) < 1e-4)
		th(3) = 0;
	if( fabs( th(4)) < 1e-4)
		th(4) = 0;
	if( fabs( th(5)) < 1e-4)
		th(5) = 0;
	if( th(3) <= 0 && th(4) <= 0 && th(5) <= 0)
	{
		J1 = pcl::PointXYZ(u,v,w);
		J2 = pcl::PointXYZ(-u,-v,-w);

		double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	v = -v; w = -w;
		}
	}
	if( th(3) <= 0 && th(4) >= 0 && th(5) >= 0)
	{
		J1 = pcl::PointXYZ(u,v,-w);
		J2 = pcl::PointXYZ(-u,-v,w);
	
		double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	v = -v;
		}
		else
			w = -w;		
			
	}
	if( th(3) >= 0 && th(4) >= 0 && th(5) <= 0)
	{
		J1 = pcl::PointXYZ(u,-v,w);
		J2 = pcl::PointXYZ(-u,v,-w);
			double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
		{
			u = -u;	w = -w;
		}
		else
			v = -v;
	
	}
	if( th(3) >= 0 && th(4) <= 0 && th(5) >= 0)
	{
		J1 = pcl::PointXYZ(u,-v,-w);
		J2 = pcl::PointXYZ(-u,v,w);
			double a1, a2;
		a1 = angle_bt_vectors( J0, J1, Pi);		
		a2 = angle_bt_vectors( J0, J2, Pi);		

		if( a1 > a2)
			u = -u;
		else
		{
			v = -v;	w = -w;
		}
	}
}


void EigenVector_Estimation( pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::ModelCoefficients::Ptr& coeff)
{
	pcl::ScopeTime t ("EigenVector Estimation");
	const double radius = 1.0;
	const double L = 5.0;
 
	Eigen::VectorXd th(6);
	double u = 0, v = 0, w = 1;
	double u1, v1, w1;
	int cloud_size = pc->size();
 
	pcl::PointXYZ J0(0.0,0.0,0.0);
	srand((unsigned)time(NULL));

	Eigen::MatrixXd A(6, 6);
	A =  Eigen::MatrixXd::Zero(6,6);
	cout<<A<<endl;

	int k;
    pcl::PointXYZ Pi;
	for( k = 0; k < 6; k++)
	{
		double random = (double)(rand()) / (RAND_MAX + 1.0);
		unsigned int index = static_cast<unsigned int>(random*cloud_size);
		Pi = pc->points[index];
		
		double a, b, c;
		a = Pi.x - J0.x;
		b = Pi.y - J0.y;
		c = Pi.z - J0.z;
	
		random = (double)(rand()) / (RAND_MAX + 1.0);
		index = static_cast<unsigned int>(random*cloud_size);
		Pi = pc->points[index];
		
		double a1, b1, c1;
		a1 = Pi.x - J0.x;
		b1 = Pi.y - J0.y;
		c1 = Pi.z - J0.z;
		
		A(k, 0) = a*a - a1*a1;
		A(k, 1) = b*b - b1*b1;
		A(k, 2) = c*c - c1*c1;
		A(k, 3) = a*b - a1*b1;
		A(k, 4) = b*c - b1*c1;
		A(k, 5) = a*c - a1*c1;	
	}

	cout<<A<<endl;
	Eigen::EigenSolver<Eigen::MatrixXd> es(A);

	cout<<"eigenvalue:"<<endl;
	cout<<es.eigenvalues()<<endl;

	cout<<"eigenvector:"<<endl;
	cout<<es.eigenvectors()<<endl;

}
