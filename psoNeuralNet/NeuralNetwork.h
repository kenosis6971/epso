/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   NeuralNetwork.h
 * Author: min
 *
 * Created on July 19, 2016, 12:00 PM
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <cstdlib>
#include "Random.h"


class NeuralNetwork {
    //private static Random rnd; // for BP to initialize wts, in PSO 
private:
    int numInput;
    int numHidden;
    int numOutput;
    double* inputs;
    double** ihWeights; // input-hidden
    double* hBiases;
    double* hOutputs;
    double** hoWeights; // hidden-output
    double* oBiases;
    double* outputs;  
    
public:
    NeuralNetwork(int numInput, int numHidden, int numOutput);
    ~NeuralNetwork();
    
    void SetWeights(double* weights, int len);
    double* GetWeights();
    double* ComputeOutputs(double* xValues, int len);
    double* Train(double ** trainData, int sampleCount, int numParticles, int maxPochs, double exitError, 
                double probDeath);
    
    double MeanSquaredError(double** trainData, int sampleCount, double* weights, int numWeights);
    double Accuracy(double ** testData, int sampleCount);
    
    static double** MakeMatrix(int rows, int cols);
    
private:
    static void Shuffle(int* sequence, int len, Random *rnd);
    static double HyperTanFunction(double x);
    static double* Softmax(double * oSum, int len);
    static int MaxIndex(double * vector, int len);
};


#endif /* NEURALNETWORK_H */

