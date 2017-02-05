#ifndef L_NET_H
#define L_NET_H

#include<vector>

using namespace std;

struct data_tuple{
	double in_one;
	double in_two;
	double in_three;
	data_tuple() {};
	data_tuple(double one,double two,double three){in_one = one,in_two = two,in_three = three;}
	data_tuple(double one,double two){in_one = one,in_two = two,in_three = 0;}
	~data_tuple() {};
};

class L_net {
	private:
		double l_rate;
		vector<vector<vector<double> > > layer_weights;
		vector<double> out_weights;
		vector<vector<double> > h_out;
		vector<vector<double> > sigma_out;
		vector<vector<double> > delta_val;
	public:
		L_net(int layers,string num_per_layer,int prob_num,float lrn_rte);
		~L_net() {};
		void trainNet(string tr_file_name,int prob_num);
		void validateNet(string v_file_name,int e_num,int prob_num);
		void testNet(string ts_file_name, int prob_num);
};

#endif