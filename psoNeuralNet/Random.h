/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Random.h
 * Author: min
 *
 * Created on July 19, 2016, 2:11 PM
 */

#ifndef RANDOM_H
#define RANDOM_H

class Random {
private:
    int min;
    int max;
    int range;
public:
    Random(int seed);
    virtual ~Random();
    double NextDouble();
    int Next(int low, int hi);
private:

};

#endif /* RANDOM_H */

