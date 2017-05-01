/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include "map.h"
#include <functional>

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
};



class ParticleFilter {
	
	// Number of particles to draw
	int num_particles; 
	
	// Internal control noise constants
	double velocity_noise;
	double yaw_rate_noise;

	
	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles
	std::vector<double> weights;

  // Transform observation to map coordinates.
  LandmarkObs observation_to_map(LandmarkObs &obs, Particle x);
	
public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param M Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param map Vector map landmarks
	 * @param transformed_observations Vector of transformed landmark observations
	 */
  void dataAssociation(std::vector<Map::single_landmark_s> map, std::vector<LandmarkObs> &transformed_observations);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations,
			Map map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();
	
	/*
	 * write Writes particle positions to a file.
	 * @param filename File to write particle positions to.
	 */
	void write(std::string filename);
	
	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}


  double calculate_weight(std::vector<LandmarkObs> vector, std::vector<LandmarkObs> observations, Map map);

  double
  calculate_weight(Particle particle, std::vector<LandmarkObs> observations,
                     std::vector<LandmarkObs> map_observations, Map map, double pDouble[]);

  LandmarkObs map_to_observation(Map::single_landmark_s &s, Particle particle);

  double calc_prob(double x, double y, double x1, double y1, double d, double d1);
};



#endif /* PARTICLE_FILTER_H_ */
