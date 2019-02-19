functions{
  
  // define ODE system for two compartmnt model
  real[] twoCptModelODE(real t,
			real[] x,
			real[] parms,
			real[] rate,  // in this example, rate is treated as data
			int[] dummy){
			  
    // Parameters
    real CL = parms[1];
    real Q = parms[2];
    real V1 = parms[3];
    real V2 = parms[4];
    real ka = parms[5];
    
    // Re-parametrization
    real k10 = CL / V1;
    real k12 = Q / V1;
    real k21 = Q / V2;
    
    // Return object (derivative)
    real y[3];  // 1 element per compartment of
                // the model

    // PK component of the ODE system
    y[1] = -ka*x[1];
    y[2] = ka*x[1] - (k10 + k12)*x[2] + k21*x[3];
    y[3] = k12*x[2] - k21*x[3];

    return y;
  }
}
data{
  int<lower = 1> np;            /* population size */
  int<lower = 1> nt;  // number of events
  int<lower = 1> nObs;  // number of observations
  int<lower = 1> iObs[nObs];  // index of observation
  
  // NONMEM data
  int<lower = 1> cmt[np * nt];
  int evid[np * nt];
  int addl[np * nt];
  int ss[np * nt];
  real amt[np * nt];
  real time[np * nt];
  real rate[np * nt];
  real ii[np * nt];
  
  vector<lower = 0>[nObs] cObs[np];  // observed concentration (dependent variable)
}

transformed data {
  vector[nObs] logCObs[np];
  int<lower = 1> len[np];
  int<lower = 1> len_theta[np];
  int<lower = 1> len_biovar[np];
  int<lower = 1> len_tlag[np];

  int nTheta = 5;   // number of parameters
  int nCmt = 3;   // number of compartments
  real biovar[np * nt, nCmt];
  real tlag[np * nt, nCmt];

  for (id in 1:np) {
    logCObs[id] = log(cObs[id]);
    for (j in 1:nt) {
      for (i in 1:nCmt) {
        biovar[(id - 1) * nt + j, i] = 1;
        tlag[(id - 1) * nt + j, i] = 0;
      }
    }
    len[id] = nt;
    len_theta[id] = nt;
    len_biovar[id] = nt;
    len_tlag[id] = nt;
  }
}

parameters{
  real<lower = 0> CL[np];
  real<lower = 0> Q[np];
  real<lower = 0> V1[np];
  real<lower = 0> V2[np];
  real<lower = 0> ka[np];
  real<lower = 0> sigma[np];
}

transformed parameters{
  real theta[np * nt, nTheta];
  vector<lower = 0>[nt] cHat[np];
  vector<lower = 0>[nObs] cHatObs[np];
  matrix<lower = 0>[nt, 3] x[np]; 

  for (id in 1:np) {
    for (it in 1:nt) {
      theta[(id - 1) * nt + it, 1] = CL[id];
      theta[(id - 1) * nt + it, 2] = Q[id];
      theta[(id - 1) * nt + it, 3] = V1[id];
      theta[(id - 1) * nt + it, 4] = V2[id];
      theta[(id - 1) * nt + it, 5] = ka[id];
    }
  }

  /* 
     pop_pk_generalOdeModel_bdf takes in the ODE system, the number of compartment 
     (here we have a two compartment model with first order absorption, so
     three compartments), the parameters matrix, the NONEM
     data, and the length of data for each individual. There
     are 4 length arrays: len for NONMEN data, len_theta for parameters,
     len_biovar for biovariablity, len_tlag for lag time.
     The user can choose between the Bdf, the Adams and the RK45 integrator.
     Returns a vector of matrices with each matrix entry
     corresponding to each individual.
  */

  x = pop_pk_generalOdeModel_bdf(twoCptModelODE, 3, len,
                                 time, amt, rate, ii, evid, cmt, addl, ss,
                                 len_theta, theta,
                                 len_biovar, biovar,
                                 len_tlag, tlag);

  for (id in 1:np) {
    cHat[id] = col(x[id], 2) ./ V1[id];
  }

  for (id in 1:np) {
    for(i in 1:nObs){
      cHatObs[id][i] = cHat[id][iObs[i]];  // predictions for observed data records
    }
  }
}

model{
  // informative prior
  for(id in 1:np){
    CL[id] ~ lognormal(log(10), 0.25);
    Q[id] ~ lognormal(log(15), 0.5);
    V1[id] ~ lognormal(log(35), 0.25);
    V2[id] ~ lognormal(log(105), 0.5);
    ka[id] ~ lognormal(log(2.5), 1);
    sigma[id] ~ cauchy(0, 1);

    logCObs[id] ~ normal(log(cHatObs[id]), sigma[id]);
  }
}

generated quantities{
  real cObsPred[np, nObs];

  for (id in 1:np) {
    for(i in 1:nObs){
      cObsPred[id, i] = exp(normal_rng(log(cHatObs[id][i]), sigma[id]));
    }
  }
			 
}
