//
// Created by thiago on 01/06/17.
//

#ifndef NEURAL_NETWORK_NEURAL_NET_H
#define NEURAL_NETWORK_NEURAL_NET_H

typedef struct neuron{
	double output;
	int input_number;
	double *input_weights;
	double error;
} neuron;

int layer_num;
int *neuron_num;
neuron **neural_net;

void create_neural_net(int layer_count, int *neuron_count);
void run(double *input);
void compute_output(neuron *n, int layer);
void backpropagate(double *target);
double* get_network_output();
double sigmoid(double x);
double get_random_weight();

#endif //NEURAL_NETWORK_NEURAL_NET_H
