/* ***Important Notes***
>the num_layers and neuro_per_layer does not account for the input and output. Output
	will always be 1 and input is prob_num + 1;
>

*/
#include<vector>
#include<iostream>
#include<cstdlib>
#include "L_net.h"

using namespace std;

int main(int argc, char* argv[]){
	if(argc != 9){
		cout << "Usage: Learn_sim num_layers neurons_per_layer learning_rate training.txt validate.txt test.txt ";
		cout << "number_epochs problem_num" << endl;
		exit(0);
	}
	int num_layers = atoi(argv[1]);
	string neuro_per_layer = argv[2];		//passed as string of ints, later subdivided and recast as individual ints ex: "2334541"
	float lrn_rte = atof(argv[3]);
	string train_file = argv[4];
	string valid_file = argv[5];
	string test_file = argv[6];
	int num_epochs = atoi(argv[7]);
	int prob_num = atoi(argv[8]);
	L_net AI_net(num_layers,neuro_per_layer,prob_num,lrn_rte);
	for(int i = 0;i < num_epochs;i++){
		AI_net.trainNet(train_file,prob_num);
		AI_net.validateNet(valid_file,i,prob_num);
	}
		AI_net.testNet(test_file,prob_num);
	//cout << "Num_layers: " << num_layers << " neuros_per_layer: " << neuro_per_layer << " learn rate: " << lrn_rte;
	//cout << " train_file: " << train_file << " valid_file: " << valid_file << " test_file: " << test_file << " epoch_num: ";
	//cout << num_epochs << " prob_num: " << prob_num << endl;	
	return 0;
}// END OF MAIN