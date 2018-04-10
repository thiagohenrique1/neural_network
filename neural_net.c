//
// Created by thiago on 01/06/17.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "neural_net.h"

void create_neural_net(int layer_count, int *neuron_count) {
	neuron_num = neuron_count;
	layer_num = layer_count;
	neural_net = malloc(layer_num * sizeof (*neural_net)) ;

	srand((unsigned int) time(NULL));

	for(int i = 0; i < layer_num; ++i){
		neural_net[i] = malloc(neuron_num[i] * sizeof(neuron));

		for (int j = 0; j < neuron_num[i]; ++j) {
			neuron* n = &(neural_net[i][j]);

			int is_bias = (i > 0) && (i < layer_num - 1) && (j == neuron_num[i] - 1);
			if(is_bias) n->output = 1;

			if(i == 0 || is_bias) n->input_number = 0;
			else n->input_number = neuron_num[i-1];
			n->input_weights = malloc(n->input_number * sizeof(*n->input_weights));

			for (int k = 0; k < n->input_number; ++k) {
				n->input_weights[k] = get_random_weight();
			}
		}
	}
}

void run(double *input){
	for (int i = 0; i < neuron_num[0]; ++i) {
		neural_net[0][i].output = input[i];
		neural_net[0][i].error = 0;
	}

	for (int i = 1; i < layer_num; ++i) {
		for (int j = 0; j < neuron_num[i]; ++j) {
			neuron *n = &neural_net[i][j];
			if(n->input_number != 0) compute_output(n, i);
			neural_net[i][j].error = 0;
//			printf("\nNeuronio %d da camada %d:\n",j,i);
//			printf("Saida: %f\n",n->output);
		}
	}
}

void compute_output(neuron *n, int layer) {
	double input_sum = 0;
	for (int i = 0; i < n->input_number; ++i) {
		input_sum += neural_net[layer-1][i].output * n->input_weights[i];
//		printf("entrada do %d: %f\n",i,neural_net[layer-1][i].output * n->input_weights[i]);
	}
	n->output = sigmoid(input_sum);
}

void backpropagate(double *target) {
	double rate = 0.2;
	for (int i = 0; i < neuron_num[layer_num-1]; ++i) {
		neuron *n = &neural_net[layer_num-1][i];
		n->error = target[i] - n->output;
	}

	for (int i = layer_num-1; i > 0; --i) {
		for (int j = 0; j < neuron_num[i]; ++j) {
			neuron *n = &neural_net[i][j];
			n->error = n->error * n->output * (1 - n->output);

			for (int k = 0; k < n->input_number; ++k) {
				neuron *back = &neural_net[i-1][k];
				n->input_weights[k] += rate * n->error * back->output;
				back->error += n->error * n->input_weights[k];
			}
		}
	}
}

double *get_network_output() {
	int last_layer_num = neuron_num[layer_num - 1];
	double *output = malloc(last_layer_num * sizeof(double));

	for (int i = 0; i < last_layer_num; ++i) {
		output[i] = neural_net[layer_num-1][i].output;
	}

	return output;
}

double sigmoid(double x) {
	return  1/(1 + exp(-x));
}

double get_random_weight() {
	return (double) rand() / (double) RAND_MAX ;
}
