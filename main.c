#include <stdio.h>
#include "neural_net.h"

int main() {
	int neuron_count[4] = {2,5,4,1};
	create_neural_net(4, neuron_count);
	double inputs[4][2];
	double target[4];
    int data_num = 4;
	int input_size = 2;
	int target_size = 1;

	FILE *file = fopen("input.txt","r");
	if(!file) return 1;

	char buff[7];
	for (int i = 0; i < data_num; ++i) {
		fgets(buff,7,file);
		char *ch = buff;
		for (int j = 0; j < input_size; ++j) {
			while (*ch == ',') ch++;
			inputs[i][j] = *ch -'0';
		}
		for (int j = 0; j < target_size; ++j) {
			while (*ch == ',') ch++;
			target[i] = *ch -'0';
		}
	}

	for (int j = 0; j < 1000000; ++j) {
		for (int i = 0; i < 4; ++i) {
			run(inputs[i]);
			backpropagate(&target[i]);
		}
	}

	printf("\nSaidas:\n");
	for (int i = 0; i < 4; ++i) {
		run(inputs[i]);
		double *out = get_network_output();
		printf("Entrada: %f %f - Saida: %f\n",inputs[i][0],inputs[i][1],*out);
	}

	return 0;
}