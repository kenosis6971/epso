/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Particle.h
 * Author: min
 *
 * Created on July 19, 2016, 1:18 PM
 */

#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
  public:
    double* position; // equivalent to NN weights
    double error; // measure of fitness
    double* velocity;

    double* bestPosition; // best position found so far by this Particle
    double bestError;
    
    //public double age; // optional used to determine death-birth

public:

    Particle(double *position, double error, double* velocity,
            double* bestPosition, double bestError, int dim);
    ~Particle();
   
}; // class Particle



#endif /* PARTICLE_H */

