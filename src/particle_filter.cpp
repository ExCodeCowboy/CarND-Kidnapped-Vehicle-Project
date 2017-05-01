/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <functional>
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void print_particles(string label, vector<Particle> particles) {
  for (int i = 0; i < particles.size(); ++i) {
    // Print your samples to the terminal.
    cout << label << particles[i].id << " " << particles[i].x << " " << particles[i].y
         << " " << particles[i].theta << " " << particles[i].weight <<endl;
  }
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  //Setup distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //Generate particles
  default_random_engine gen;
  int particles_to_make = 10;
  particles = vector<Particle>((ulong)particles_to_make);

  for (int i = 0; i < particles_to_make; ++i) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

    particles[i] = Particle{id:i, x:sample_x, y:sample_y, theta:sample_theta, weight:1.0/particles_to_make};
	}

  print_particles("Sample", particles);

	//Initialize state variables;
  num_particles = particles_to_make;
	is_initialized = true;

  //Initialize control noise
  velocity_noise = 10.0;
  yaw_rate_noise = 0.05;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


  //Setup distributions / ranged around 0 so I can reuse them (maybe multiply by delta_t?)
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  default_random_engine gen;

  //Predict new particle positions w/ control noise
  for (int i = 0; i < particles.size(); ++i) {
    Particle &particle = particles[i];

    double yaw = particle.theta;
    double v = velocity ;//+ dist_velocity(gen);
    double yaw_rate = yaw_rate;// + dist_yaw_rate(gen);

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
    particle.x += n_px + dist_x(gen);
    particle.y += n_py + dist_y(gen);
    particle.theta = fmod(particle.theta + n_yaw + dist_theta(gen), 2 * M_PI);

  }
  //Copy into working particles
  print_particles("Predict", particles);
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
    cout << "Transformed:" << observation.x << " " << observation.y << " map:" << map[observation.id-1].x_f << " " <<
         map[observation.id-1].y_f << endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  //Optimization idea: pre-process landmarks to a quad tree to find matches faster.
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
  }

  print_particles("Weighted", particles);
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html



}

float get_random()
{
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}

void ParticleFilter::resample() {
  //Setup base measures
  double biggest = 0;
  for (int i = 0; i < particles.size(); ++i) {
    Particle &particle = particles[i];
    if (particle.weight > biggest) {
      biggest = particle.weight;
      cout<< particle.weight << " weight "<< endl;
    }
  }

  vector<Particle> sampled_particles = vector<Particle>(particles.size());
  //set random startpos
  ulong pos = 0;
  for (int i = 0; i < particles.size(); ++i) {
    long double step = biggest * 2.0 * get_random();
    while ((step > 0.0) && (step > particles[pos].weight)) {
      step = step - particles[pos].weight;
      pos = (pos + 1) % particles.size();
      //cout<< step << " step "<< endl;
    }
    cout<< "picked "<< pos << endl;
    Particle sampled = Particle();
    sampled.id = i;
    sampled.x = 0.0 + particles[pos].x;
    sampled.y = 0.0 + particles[pos].y;
    sampled.weight = 0.0 + particles[pos].weight;
    sampled.theta = 0.0 + particles[pos].theta;
    sampled_particles[i] = sampled;
  }
  particles = sampled_particles;

  // TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
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
  double map_x = particle.x + (obs.x*cos(particle.theta)) + (obs.y * sin(particle.theta));
  double map_y = particle.y + (obs.x*sin(particle.theta)) + (obs.y * cos(particle.theta));
  int old_id = obs.id;
  LandmarkObs result = LandmarkObs();
  result.x = map_x;
  result.y = map_y;
  result.id = old_id;
   return result;
}

double ParticleFilter::calculate_weight(Particle particle, vector<LandmarkObs> observations,
                                        vector<LandmarkObs> map_observations, Map map, double std_landmark[]) {
  double prob = 1.0;
  for (int pos = 0; pos < observations.size(); ++pos) {
    LandmarkObs &observation = observations[pos];
    int landmark_id = map_observations[pos].id;
    Map::single_landmark_s &landmark = map.landmark_list[landmark_id-1]; //position is one less.
    LandmarkObs map_landmark = map_to_observation(landmark, particle);
    cout << "Original:" << observation.x << " " << observation.y << " MapTransformed:" << map_landmark.x << " " << map_landmark.y << endl;
    prob = prob * calc_prob(observation.x, observation.y, map_landmark.x, map_landmark.y, std_landmark[0],
                            std_landmark[1]);
  }
  return prob;
}

LandmarkObs ParticleFilter::map_to_observation(Map::single_landmark_s &landmark, Particle particle) {
  double obs_x = ((landmark.x_f-particle.x)*cos(-particle.theta)) + ((landmark.y_f-particle.y) * sin(-particle.theta));
  double obs_y = ((landmark.x_f-particle.x)*sin(-particle.theta)) + ((landmark.y_f-particle.y) * cos(-particle.theta));
  LandmarkObs landmarkObs = LandmarkObs();
  landmarkObs.x = obs_x;
  landmarkObs.y = obs_y;
  return landmarkObs;
}

double ParticleFilter::calc_prob(double x, double y, double xm, double ym, double dev_x, double dev_y) {
  double xPart = pow(x - xm, 2)/(2*pow(dev_x, 2));
  double yPart = pow(y - ym, 2)/(2*pow(dev_y, 2));
  return (1.0/(2.0*M_PI*dev_x*dev_y))*exp(-(xPart+yPart));
}
