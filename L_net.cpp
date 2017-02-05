#include<iostream>
#include<cstddef>
#include<iostream>
#include<cmath>
#include<fstream>
#include<cstdlib>
#include<ctime>
#include<string>
#include "L_net.h"

L_net::L_net(int layers,string num_per_layer,int prob_num,float lrn_rte){
		int i,j,k;
		double int_state;
		srand(time(NULL));
		layer_weights.resize(layers);
		h_out.resize(layers);
		sigma_out.resize(layers);
		delta_val.resize(layers);
		string num_neurons = "";
		int n_neurons = 0;
		int prev_n = 0;
		l_rate = lrn_rte;
		int bitpiece = 0;
		for(i = 0;i < layers;i++){
			num_neurons = num_per_layer.substr(bitpiece,2);
			n_neurons = atoi(num_neurons.c_str());
			//cout << n_neurons << endl;
			layer_weights[i].resize(n_neurons);
			h_out[i].resize(n_neurons);
			sigma_out[i].resize(n_neurons);
			delta_val[i].resize(n_neurons);
			bitpiece += 2;
			if(i == 0){
				for(j = 0;j < layer_weights[i].size();j++){
					layer_weights[0][j].resize(prob_num + 1);
				}
			}else{
				for(j = 0;j < layer_weights[i].size();j++){
					layer_weights[i][j].resize(prev_n + 1);			// Add + 1 for threshold
				}
				if(i == layers -1){
					out_weights.resize(layer_weights[i].size() + 1);
				}
			}
			num_neurons = "";
			prev_n = n_neurons;
			n_neurons = 0;
		}
		/*
		for(i = 0;i < layers;i++){
			cout << "Layer: " << i << " num_neurons: " << layer_weights[i].size() << endl;
			for(j = 0;j < layer_weights[i].size();j++){
				cout << "     Neuron: " << j << " num_weights: " << layer_weights[i][j].size() << endl;
			}
		}
		*/
		double r_num;
		int r_sign;
		double n_weight;
	for(i = 0;i < layer_weights.size();i++){
		for(j = 0;j < layer_weights[i].size();j++){
			for(k = 0;k < layer_weights[i][j].size();k++){			// Technically k = 0 is input values, but we change it later anyway so randomly
				r_num = (double)rand() / RAND_MAX;					// giving it an initial value doesn't matter.
				n_weight = r_num * (.1);
				r_sign = rand() % 2;
				if(r_sign == 1) {n_weight *= -1;}
				layer_weights[i][j][k] = n_weight;
				//cout << "layer: " << i << " node: " << j << " weight: " << n_weight << endl;
			}
		}
	}
	for(i = 0;i < out_weights.size();i++){
		r_num = (double)rand() / RAND_MAX;
		n_weight = r_num * .1;
		r_sign = rand() % 2;
		if(r_sign == 1) {n_weight *= -1;}
		out_weights[i] = n_weight;
	}
	for(i = 0;i < h_out.size();i++){
		for(j = 0;j < h_out[i].size();j++){
			h_out[i][j] = 0;
			sigma_out[i][j] = 0;
			delta_val[i][j] = 0;
		}
	}
}

void L_net::trainNet(string tr_file_name,int prob_num){
	int i,j,k,m;
	ifstream fin;
	fin.open(tr_file_name.c_str());
	if(!fin.is_open()){
		cout << "Error opening read file in train method, program terminated." << endl;
		exit(0);
	}
	vector<data_tuple> inputs;
	inputs.resize(200);
	vector<double> e_vals;
	e_vals.resize(200);
	double cur_h,final_h,final_sigma,first_delta,cur_delta,prev_delta;
	for( i = 0;i < 200;i++){
		data_tuple n_data;
		double d1,d2,d3;
		if(prob_num == 1){
			fin >> d1 >> d2;
			n_data = data_tuple(d1,d2);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
		if(prob_num == 2){
			fin >> d1 >> d2 >> d3;
			n_data = data_tuple(d1,d2,d3);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
	}
	for(m = 0;m < 200;m++){
		/* RESET ALL DELTA, SIGMA, AND H VALS*/
		for(i = 0;i < h_out.size();i++){
			for(j = 0;j < h_out[i].size();j++){
				h_out[i][j] = 0;
				sigma_out[i][j] = 0;
				delta_val[i][j] = 0;
			}
		}
		/* BEGIN FORWARD PROPOGATION HERE */
		for(i = 0;i < layer_weights.size();i++){
			for(j = 0;j < layer_weights[i].size();j++){
				if(i == 0){
					for(k = 0;k < layer_weights[i][j].size();k++){
						if(prob_num == 2){
							h_out[i][j] += layer_weights[i][j][0] * inputs[m].in_one;
							h_out[i][j] += layer_weights[i][j][1] * inputs[m].in_two;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							//h_out[i][j] += cur_h;
						}
						if(prob_num == 3){
							h_out[i][j] += layer_weights[i][j][0] * inputs[m].in_one;
							h_out[i][j] += layer_weights[i][j][1] * inputs[m].in_two;
							h_out[i][j] += layer_weights[i][j][2] * inputs[m].in_three;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_three;
							//h_out[i][j] += cur_h;
						}
					}
					//cur_h = 0;
					sigma_out[i][j] = (1/(1+exp(h_out[i][j]*-1)));
				}else{
					for(k = 0;k < layer_weights[i-1].size();k++){	//k = 1
						h_out[i][j] += layer_weights[i][j][k+1] * sigma_out[i-1][k];
						//cur_h = layer_weights[i][j][k] * sigma_out[i-1][k-1];
						//h_out[i][j] += cur_h;
						//cur_h = 0;
					}
					sigma_out[i][j] = (1/(1+exp((h_out[i][j] + layer_weights[i][j][0])*-1)));
				}
			}
		}
		final_h = 0;
		final_sigma = 0;
		for(i = 0;i < out_weights.size()-1;i++){		// i = 1
			final_h += out_weights[i+1] * sigma_out[layer_weights.size()-1][i];
			//cur_h = out_weights[i] * sigma_out[layer_weights.size()-1][i-1];
			//final_h += cur_h;
		}
		final_sigma = (1/(1+exp((final_h + out_weights[0])*-1)));
		for(i = 0;i < sigma_out.size();i++){
			for(j = 0;j < sigma_out[i].size();j++){
				//cout << sigma_out[i][j] << endl;
			}
		}
		/* BEGIN BACK PROPOGATION HERE */
		
		first_delta = final_sigma * (1-final_sigma) * (e_vals[m]-final_sigma);
		//cur_delta = 0;
		for(i = layer_weights.size() -1;i >= 0;i--){
			if(i == layer_weights.size() - 1){
				for(j = 0;j < layer_weights[i].size();j++){
					 //cout << sigma_out[i][j] << endl;
					delta_val[i][j] = sigma_out[i][j] * (1-sigma_out[i][j]) * first_delta * out_weights[j+1];
					//cur_delta = sigma_out[i][j] * (1-sigma_out[i][j]) * first_delta * out_weights[j+1];
					//delta_val[i][j] = cur_delta; // This line...this bloody line caused so much trouble
					//cout << delta_val[i][j] << endl;
				}
			}else{
				for(j = 0;j < layer_weights[i].size();j++){
					for(k = 0;k < layer_weights[i+1].size();k++){
						cur_delta = 0;
						//cout << sigma_out[i][j] << endl;
						//delta_val[i][j] += sigma_out[i][j] * (1-sigma_out[i][j]) * delta_val[i+1][k] * layer_weights[i+1][[k][j+1];
						cur_delta = sigma_out[i][j] * (1-sigma_out[i][j]) * delta_val[i+1][k] * layer_weights[i+1][k][j+1];
						delta_val[i][j] += cur_delta;
						cur_delta = 0;
					}
					//cout << delta_val[i][j] << endl;
				}
			}
			//cur_delta = 0;
		}
		
		/* BEGIN WEIGHT UPDATE HERE */
		for(i = 0;i < layer_weights.size();i++){
			for(j = 0;j < layer_weights[i].size();j++){
				if(i == 0){
					for(k = 0;k < layer_weights[i][j].size();k++){
						if(prob_num == 2){
							layer_weights[i][j][0] += l_rate * delta_val[i][j] * inputs[m].in_one;
							layer_weights[i][j][1] += l_rate * delta_val[i][j] * inputs[m].in_two;
						}
						if(prob_num == 3){
							layer_weights[i][j][0] += l_rate * delta_val[i][j] * inputs[m].in_one;
							layer_weights[i][j][1] += l_rate * delta_val[i][j] * inputs[m].in_two;
							layer_weights[i][j][2] += l_rate * delta_val[i][j] * inputs[m].in_three;
						}
					}
				}else{
					for(k = 1;k < layer_weights[i][j].size();k++){
						layer_weights[i][j][k] += l_rate * delta_val[i][j] * sigma_out[i-1][k-1];
						//cout << sigma_out[i-1][k-1] << endl;
					}
					layer_weights[i][j][0] += l_rate * delta_val[i][j];
				}
			}
		}
		for(i = 1;i < out_weights.size();i++){
			out_weights[i] += l_rate * first_delta * sigma_out[layer_weights.size()-1][i-1];
		}
		out_weights[0] += l_rate * first_delta;
		//cout << layer_weights[1][1][1] << endl;
	}
	//cout << "got out" << endl;
	/*
	cout << "train test: " << endl;
	for(int j = 0;j < 200;j++){
		if((j % 50) == 0){
			cout << inputs[j].in_one << " " << inputs[j].in_two << " " << inputs[j].in_three << " " << e_vals[j] << endl;
		}
	}
	*/

}

void L_net::validateNet(string v_file_name,int e_num,int prob_num){
	//cout << "validating" << endl;
	int i,j,k,m;
	ifstream fin;
	fin.open(v_file_name.c_str());
	if(!fin.is_open()){
		cout << "Error opening read file in validate method, program terminated." << endl;
		exit(0);
	}
	vector<data_tuple> inputs;
	inputs.resize(100);
	vector<double> e_vals;
	e_vals.resize(100);
	double cur_h,final_h,final_sigma,sum,rmse;
	for(i = 0;i < 100;i++){
		data_tuple n_data;
		double d1,d2,d3;
		if(prob_num == 1){
			fin >> d1 >> d2;
			n_data = data_tuple(d1,d2);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
		if(prob_num == 2){
			fin >> d1 >> d2 >> d3;
			n_data = data_tuple(d1,d2,d3);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
	}	
	/* BEGIN FORWARD PROPOGATION HERE */
	rmse = 0;
	sum = 0;
	for(m = 0;m < 100;m++){
		/* RESET ALL DELTA, SIGMA, AND H VALS*/
		for(i = 0;i < h_out.size();i++){
			for(j = 0;j < h_out[i].size();j++){
				h_out[i][j] = 0;
				sigma_out[i][j] = 0;
				delta_val[i][j] = 0;
			}
		}
		/* BEGIN FORWARD PROPOGATION HERE */
		for(i = 0;i < layer_weights.size();i++){
			for(j = 0;j < layer_weights[i].size();j++){
				if(i == 0){
					for(k = 0;k < layer_weights[i][j].size();k++){
						if(prob_num == 2){
							h_out[i][j] += layer_weights[i][j][0] * inputs[m].in_one;
							h_out[i][j] += layer_weights[i][j][1] * inputs[m].in_two;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							//h_out[i][j] += cur_h;
						}
						if(prob_num == 3){
							h_out[i][j] += layer_weights[i][j][0] * inputs[m].in_one;
							h_out[i][j] += layer_weights[i][j][1] * inputs[m].in_two;
							h_out[i][j] += layer_weights[i][j][2] * inputs[m].in_three;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							//h_out[i][j] += cur_h;
							//cur_h = layer_weights[i][j][k] * inputs[m].in_three;
							//h_out[i][j] += cur_h;
						}
					}
					//cur_h = 0;
					sigma_out[i][j] = (1/(1+exp(h_out[i][j]*-1)));
				}else{
					for(k = 0;k < layer_weights[i-1].size();k++){	//k = 1
						h_out[i][j] += layer_weights[i][j][k+1] * sigma_out[i-1][k];
						//cur_h = layer_weights[i][j][k] * sigma_out[i-1][k-1];
						//h_out[i][j] += cur_h;
						//cur_h = 0;
					}
					sigma_out[i][j] = (1/(1+exp((h_out[i][j] + layer_weights[i][j][0])*-1)));
				}
			}
		}
		final_h = 0;
		final_sigma = 0;
		for(i = 0;i < out_weights.size()-1;i++){		// i = 1
			final_h += out_weights[i+1] * sigma_out[layer_weights.size()-1][i];
			//cur_h = out_weights[i] * sigma_out[layer_weights.size()-1][i-1];
			//final_h += cur_h;
		}
		final_sigma = (1/(1+exp((final_h + out_weights[0])*-1)));
		sum += (e_vals[m] + final_sigma) * (e_vals[m] + final_sigma);
	}
	
	rmse = sqrt(sum/200);		//sqrt((1/(2*100))*sum);
	if(e_num % 1000 == 0){
		cout << "first weights: " << layer_weights[0][0][0] << "  " << layer_weights[0][0][1] << endl;
		cout << "final weights: " << out_weights[0] << "  " << out_weights[1] << "  " << out_weights[2] << endl;
		cout << "final_sigma: " << final_sigma << endl;
		cout << "inputs: " << inputs[m-1].in_one << "  " << inputs[m-1].in_two << endl;
		cout << "expected ans: " << e_vals[m-1] << endl;
		cout << "sum: " << sum << endl;
		cout << "Root Mean Square Error: " << rmse << endl;
		cout << "epoch: " << e_num << endl;
	}
	
	/*
	cout << "valid test: " << endl;
	for(int j = 0;j < 100;j++){
		if((j % 25) == 0){
			cout << inputs[j].in_one << " " << inputs[j].in_two << " " << inputs[j].in_three << " " << e_vals[j] << endl;
		}
	}
	*/
}
void L_net::testNet(string ts_file_name,int prob_num){
	int i,j,k,m;
	int e_num = 0;
	ifstream fin;
	fin.open(ts_file_name.c_str());
	if(!fin.is_open()){
		cout << "Error opening read file in test method, program terminated." << endl;
		exit(0);
	}
	vector<data_tuple> inputs;
	inputs.resize(50);
	vector<double> e_vals;
	e_vals.resize(50);
	double cur_h,final_h,final_sigma,sum,rmse;
	for(i = 0;i < 50;i++){
		data_tuple n_data;
		double d1,d2,d3;
		if(prob_num == 1){
			fin >> d1 >> d2;
			n_data = data_tuple(d1,d2);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
		if(prob_num == 2){
			fin >> d1 >> d2 >> d3;
			n_data = data_tuple(d1,d2,d3);
			inputs[i] = n_data;
			fin >> e_vals[i];
		}
	}
	/* BEGIN FORWARD PROPOGATION HERE */
	sum = 0;
	rmse = 0;
	for(m = 0;m < 50;m++){
		for(i = 0;i < layer_weights.size();i++){
			for(j = 0;j < layer_weights[i].size();j++){
				if(i == 0){
					for(k = 0;k < layer_weights[i][j].size();k++){
						if(prob_num == 2){
							cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							h_out[i][j] += cur_h;
							cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							h_out[i][j] += cur_h;
						}
						if(prob_num == 3){
							cur_h = layer_weights[i][j][k] * inputs[m].in_one;
							h_out[i][j] += cur_h;
							cur_h = layer_weights[i][j][k] * inputs[m].in_two;
							h_out[i][j] += cur_h;
							cur_h = layer_weights[i][j][k] * inputs[m].in_three;
							h_out[i][j] += cur_h;
						}
					}
					cur_h = 0;
					sigma_out[i][j] = (1/(1+exp((h_out[i][j] + layer_weights[i][j][0])*-1)));
				}else{
					for(k = 1;k < layer_weights[i][j].size();k++){
						cur_h = layer_weights[i][j][k] * sigma_out[i-1][k-1];
						h_out[i][j] += cur_h;
						cur_h = 0;
					}
					sigma_out[i][j] = (1/(1+exp((h_out[i][j] + layer_weights[i][j][0])*-1)));
				}
			}
		}
		final_h = 0;
		final_sigma = 0;
		for(i = 1;i < out_weights.size();i++){
			cur_h = out_weights[i] * sigma_out[layer_weights.size()-1][i-1];
			final_h += cur_h;
		}
		final_sigma = (1/(1+exp((final_h + out_weights[0])*-1)));
		sum += (e_vals[m] + final_sigma) * (e_vals[m] + final_sigma);
	}
	e_num++;
	/*
	rmse = sqrt((1/(2*50))*sum);
	if(e_num % 1000 == 0){
		cout << "inputs: " << inputs[m-1].in_one << "  " << inputs[m-1].in_two << endl;
		cout << "expected ans: " << e_vals[m-1] << endl;
		cout << "sum: " << sum << endl;
		cout << "Root Mean Square Error: " << rmse << endl;
		cout << "epoch: " << e_num << endl;
	}
	*/
	/*
	cout << "test test: " << endl;
	for(int j = 0;j < 50;j++){
		if((j % 10) == 0){
			cout << inputs[j].in_one << " " << inputs[j].in_two << " " << inputs[j].in_three << " " << e_vals[j] << endl;
		}
	}
	*/
}