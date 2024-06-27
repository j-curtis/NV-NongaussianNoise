#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <ctime>
#include <complex>
#include <string>

using namespace std;

const double pi = 3.14159265358979323846;
	
//Measure energy in units of J
const double J = 1.;
const double dt = .01/J;

std::string bool2str(bool tf){
	if(tf){
		return "True";
	}
	else{
		return "False";
	}
}


int main() {

	//Simulation parameters
	const size_t L = 80; //Lattice size
	const size_t ntimes = 500000; //Number of time steps (run once to burn in then again to capture dynamics)
	int nburn = 100000; //We don't need to burn for an entire sample time
	nburn = std::min(int(ntimes), nburn);
	double T = 0.5*J; //Temperature

	//Correlation function parameters
	int Ns = 500; //Number of points we sample the random vorticity in order to compute the ensemble average 
	//Note this will end up being the number of succesful samples we make 

	const size_t ndt = 1; //Number of delay-time time steps we evaluate the correlation function for

	std::cout<<"dt = "<<dt<<", L = "<<L<<", nburn = "<<nburn<<", ntimes = "<<ntimes<<", T/J = "<<T<<", Nsample = "<<Ns<<", ndt out = "<<ndt<<std::endl;

	//File I/O flags
	const bool vort_out = true; //If this is true we write out the vorticity itself

	//Start timer
   	int t0 = std::time(NULL);

	//Initialize RNGs and define the various distributions we will need
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  	std::default_random_engine generator (seed);
  	std::normal_distribution<double> normal(0.0,sqrt(2.*T*dt)); //Random noise for stochastic evolution

  	std::uniform_int_distribution<int> timept(0,ntimes - ndt - 1); //Random time point for correlation function sampling procedure
  	std::uniform_int_distribution<int> spacept(0,L-1); //Random spatial point (one ordinate/abscissa) for correlation function sampling procedure 

  	//Allocate array of theta(t,x) 
   	std::vector<std::vector<std::vector<double> > > thetas(ntimes, std::vector<std::vector<double> >(L, std::vector<double>(L) ) );
	
	//BURN LOOP
	int t1 = std::time(NULL);
	//Burn loop is one run of the full simulation to ensure the initial condition is sufficiently random
	for(int nt = 1; nt < nburn; ++nt){
		for(int nx = 0; nx < L; nx++){
			for(int ny = 0; ny <L; ny++){

				//Implement the dynamics -- first the dE/dtheta terms
				thetas[nt][nx][ny] = thetas[nt-1][nx][ny] - J*dt*(
				 sin( thetas[nt-1][nx][ny] - thetas[nt-1][(nx+1)%L][ny] )
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][(nx-1)%L][ny] ) 
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][nx][(ny+1)%L] )
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][nx][(ny-1)%L] ) );

				//Now add a random noise term
				thetas[nt][nx][ny] += normal(generator);

				//Now bring back to interval [-pi,pi]
				thetas[nt][nx][ny] = std::fmod(thetas[nt][nx][ny] , 2.*pi);

			}
		}
	}
	int t2 = std::time(NULL);

	std::cout<<"Burn loop time: "<<std::to_string(t2-t1)<<"s"<<std::endl;

	//Initialize the array back with the thermalized state
	for(int nx = 0; nx < L; nx++){
		for(int ny = 0; ny <L; ny++){
			thetas[0][nx][ny] = thetas[nburn-1][nx][ny];
		}
	}

	//CAPTURE LOOP
	t1 = std::time(NULL);
	//Now we collect the data to use for the actual calculation
	for(int nt = 1; nt < ntimes; ++nt){
		for(int nx = 0; nx < L; nx++){
			for(int ny = 0; ny <L; ny++){

				//Implement the dynamics -- first the dE/dtheta terms
				thetas[nt][nx][ny] = thetas[nt-1][nx][ny] - J*dt*(
				 sin( thetas[nt-1][nx][ny] - thetas[nt-1][(nx+1)%L][ny] )
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][(nx-1)%L][ny] ) 
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][nx][(ny+1)%L] )
					+sin( thetas[nt-1][nx][ny] - thetas[nt-1][nx][(ny-1)%L] ) );

				//Now add a random noise term
				thetas[nt][nx][ny] += normal(generator);

				//Now bring back to interval [-pi,pi]
				thetas[nt][nx][ny] = std::fmod(thetas[nt][nx][ny] , 2.*pi);

			}
		}
	}
	t2 = std::time(NULL);

	std::cout<<"Capture loop done: "<<std::to_string(t2-t1)<<"s"<<std::endl;

	//EXTRACT VORTICITY
	//We probably only need to compute this Nsample times unless we want the full stochastic evolution
	//We now compute the time-traces of the vorticity
	//We define the vorticity for a site r such that v(r) = (theta(r+x)-theta(r))mod 2pi + (theta(r+x+y)-theta(r+x))mod 2pi  +(theta(r+y)-theta(r+x+y))mod 2pi  +(theta(r)-theta(r+y))mod 2pi 
   	std::vector<std::vector<std::vector<double> > > vort(ntimes, std::vector<std::vector<double> >(L, std::vector<double>(L) ) );
	
	t1 = std::time(NULL);
   	for(int t =0; t < ntimes; t++){
   		for(int x = 0; x < L; x++){
   			for(int y =0; y< L ; y++){
   				vort[t][x][y] = std::fmod(thetas[t][(x+1)%L][y] - thetas[t][x][y],2.*pi);
   				vort[t][x][y] += std::fmod(thetas[t][(x+1)%L][(y+1)%L] - thetas[t][(x+1)%L][y],2.*pi);
   				vort[t][x][y] += std::fmod(thetas[t][x][(y+1)%L] - thetas[t][(x+1)%L][(y+1)%L],2.*pi);
   				vort[t][x][y] += std::fmod(thetas[t][x][y] - thetas[t][x][(y+1)%L],2.*pi);
   			}
   		}
 	}
	t2 = std::time(NULL);


 	std::cout<<"Vorticity loop done: "<<std::to_string(t2-t1)<<"s"<<std::endl;
	
	std::cout<<"vorticity_out: "<<bool2str(vort_out)<<std::endl;
	//PRINT OUT VORTICITY
	if(vort_out){
		t1 = std::time(NULL);
		std::ofstream outfile;
		string fname = "./vorticity_L=" + std::to_string(L)+"_t="+std::to_string(ntimes)+"_T="+std::to_string(T)+".csv";
		outfile.open(fname);

		for(int nt = 0; nt < ntimes; nt++){
			for(int nx =0; nx < L; nx++){
				for(int ny = 0; ny < L; ny++){
					outfile<<" "<<vort[nt][nx][ny];
				}
			}
			outfile<<std::endl;
		}
		outfile.close();
		t2 = std::time(NULL);
		std::cout<<"vort_out loop done: "<<std::to_string(t2-t1)<<"s"<<std::endl;
	}
	
	int tf = std::time(NULL);
	std::cout<<"All done: "<<(tf-t0)<<"s"<<std::endl;

	
	return 0;
}
