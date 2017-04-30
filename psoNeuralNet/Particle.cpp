/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Particle.cpp
 * Author: min
 * 
 * Created on July 19, 2016, 1:18 PM
 */

#include "Particle.h"
#include <cstring>

Particle::Particle(double* position, double error, double* velocity,
            double* bestPosition, double bestError, int dim) {
        this->position = new double[dim];   
        memcpy(this->position, position, dim * sizeof(double));
        this->error = error;
        this->velocity = new double [dim];
        memcpy(this->velocity, velocity, dim * sizeof(double));
        this->bestPosition = new double[dim];
        memcpy(this->bestPosition, bestPosition, dim * sizeof(double));
        this->bestError = bestError;
        
        //this.age = 0;
}

Particle::~Particle() {
    delete [] position;
    delete [] velocity;
    delete [] bestPosition;
}

