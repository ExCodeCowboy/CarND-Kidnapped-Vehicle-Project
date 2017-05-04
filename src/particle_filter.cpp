/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <functional>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

//Debug output so you can see a snapshot of the current particles.
void print_particles(string label, vector<Particle> particles) {
  for (int i = 0; i < particles.size(); ++i) {
    // Print your samples to the terminal.
    cout << label << particles[i].id << " " << particles[i].x << " " << particles[i].y
         << " " << particles[i].theta << " " << particles[i].weight <<endl;
  }
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {

  //Setup distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //Generate particles
  default_random_engine gen;
  int particles_to_make = 40;
  particles = vector<Particle>((ulong)particles_to_make);

  for (int i = 0; i < particles_to_make; ++i) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

    particles[i] = Particle{id:i, x:sample_x, y:sample_y, theta:sample_theta, weight:1.0/particles_to_make};
	}

#ifdef DEBUG
  print_particles("Sample", particles);
#endif

	//Initialize state variables;
  num_particles = particles_to_make;
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  //Setup noise distributions / ranged around 0 so I can reuse them
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  //Startup a random engine
  default_random_engine gen;

  //Predict new particle positions
  for (int i = 0; i < particles.size(); ++i) {
    Particle &particle = particles[i];

    double yaw = particle.theta;
    double v = velocity ;//+ dist_velocity(gen);

    //Predict new position deltas given noisy input.
    double n_px, n_py, n_yaw;
    if (yaw_rate != 0) {
      //Non-zero
      n_px = (v/yaw_rate) * (sin(yaw + (yaw_rate * delta_t)) - sin(yaw));
      n_py = (v/yaw_rate) * (-cos(yaw + (yaw_rate * delta_t)) + cos(yaw));
      n_yaw = (yaw_rate * delta_t);
    } else {
      //Zero
      n_px = (v * cos(yaw) * delta_t);
      n_py = (v * sin(yaw) * delta_t);
      n_yaw = 0;
    }

    //Add predicted movement and noise (handle full rotations)
    particle.x += n_px + dist_x(gen);
    particle.y += n_py + dist_y(gen);
    particle.theta = fmod(particle.theta + n_yaw + dist_theta(gen), 2 * M_PI);

  }
  //Copy into working particles
#ifdef DEBUG
  print_particles("Predict", particles);
#endif
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> map, std::vector<LandmarkObs>& transformed_observations) {
  for (int opos = 0; opos < transformed_observations.size(); ++opos) {
    double best = DBL_MAX;
    LandmarkObs &observation = transformed_observations[opos];
    for (int lpos = 0; lpos < map.size(); ++lpos) {
      double dist = sqrt(pow(observation.x - map[lpos].x_f, 2) + pow(observation.y - map[lpos].y_f, 2));
      if (dist < best) {
        observation.id = map[lpos].id_i;
        best = dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  //Optimization idea: pre-process landmarks to a quadtree to find matches faster.

  //Track total weight
  double total_weight = 0;
  for (int i = 0; i < particles.size(); ++i) {
    Particle &particle = particles[i];
    //Transform observations to map space for this particle.
    vector<LandmarkObs> map_observations = vector<LandmarkObs>(observations.size());
    for (int obs = 0; obs < observations.size(); ++obs) {
      map_observations[obs] = observation_to_map(observations[obs], particle);
    }

    //Match to map
    dataAssociation(map_landmarks.landmark_list, map_observations);

    //transform both back to car space
    particle.weight = calculate_weight(particle, observations, map_observations, map_landmarks, std_landmark);
    total_weight += particle.weight;
  }

  //Normalize the weights
  if (total_weight > 0) {
    for (int i = 0; i < particles.size(); ++i) {
      Particle &particle = particles[i];
      particle.weight = particle.weight/total_weight;
    }
  }

#ifdef DEBUG
  print_particles("Weighted", particles);
#endif
}

//Random float utility function
float get_random()
{
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1
  return dis(e);
}

void ParticleFilter::resample() {
  //Find largest
  double biggest = 0;
  for (int i = 0; i < particles.size(); ++i) {
    Particle &particle = particles[i];
    if (particle.weight > biggest) {
      biggest = particle.weight;
    }
  }


  vector<Particle> sampled_particles = vector<Particle>(particles.size());
  ulong pos = (ulong) particles.size() * get_random();
  for (int i = 0; i < particles.size(); ++i) {
    long double step = biggest * 2.0 * get_random();
    while ((step > 0.0) && (step > particles[pos].weight)) {
      step = step - particles[pos].weight;
      pos = (pos + 1) % particles.size();
    }
    Particle sampled = Particle();
    sampled.id = i;
    sampled.x = particles[pos].x;
    sampled.y = particles[pos].y;
    sampled.weight = particles[pos].weight;
    sampled.theta = particles[pos].theta;
    sampled_particles[i] = sampled;
  }
  particles = sampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

LandmarkObs ParticleFilter::observation_to_map(LandmarkObs &obs, Particle particle) {
  //Translate to map coordinates from relative coordinates
  double map_x = particle.x + (obs.x*cos(particle.theta)) - (obs.y * sin(particle.theta));
  double map_y = particle.y + (obs.x*sin(particle.theta)) + (obs.y * cos(particle.theta));
  int old_id = obs.id;
  LandmarkObs result = LandmarkObs();
  result.x = map_x;
  result.y = map_y;
  result.id = old_id;
   return result;
}

LandmarkObs ParticleFilter::map_to_observation(Map::single_landmark_s &landmark, Particle particle) {
  //Translate map landmark to relative observation
  double obs_x = ((landmark.x_f-particle.x)*cos(-particle.theta)) - ((landmark.y_f-particle.y) * sin(-particle.theta));
  double obs_y = ((landmark.x_f-particle.x)*sin(-particle.theta)) + ((landmark.y_f-particle.y) * cos(-particle.theta));
  LandmarkObs landmarkObs = LandmarkObs();
  landmarkObs.x = obs_x;
  landmarkObs.y = obs_y;
  return landmarkObs;
}

double ParticleFilter::calculate_weight(Particle particle, vector<LandmarkObs> observations,
                                        vector<LandmarkObs> map_observations, Map map, double std_landmark[]) {
  //For a given particle apply probability for each observation.
  double prob = 1.0;
  for (int pos = 0; pos < observations.size(); ++pos) {
    LandmarkObs &observation = observations[pos];
    int landmark_id = map_observations[pos].id;
    Map::single_landmark_s &landmark = map.landmark_list[landmark_id-1]; //position is one less.
    LandmarkObs map_landmark = map_to_observation(landmark, particle);
    prob = prob * calc_prob(observation.x, observation.y, map_landmark.x, map_landmark.y, std_landmark[0],
                            std_landmark[1]);
  }
  return prob;
}

double ParticleFilter::calc_prob(double x, double y, double xm, double ym, double dev_x, double dev_y) {
  double xPart = pow(x - xm, 2)/(2*pow(dev_x, 2));
  double yPart = pow(y - ym, 2)/(2*pow(dev_y, 2));
  return (1.0/(2.0*M_PI*dev_x*dev_y))*exp(-(xPart+yPart));
}
