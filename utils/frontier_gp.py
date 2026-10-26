"""Objective-specific empirical-Bayes molecular GP (surrogate version 2).

Fit kernel mixture, descriptor length and noise from the supplied history only.
The fixed bounds and deterministic starting point apply to every objective and
pocket. Returns latent predictive uncertainty, without observation noise.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize


def components(a,b):
    af,bf=a[:,:2048],b[:,:2048]
    dot=af@bf.T
    tanimoto=dot/np.maximum(af.sum(1)[:,None]+bf.sum(1)[None,:]-dot,1e-12)
    distance=((a[:,None,2048:]-b[None,:,2048:])**2).sum(-1)
    return tanimoto,distance


def likelihood(parameters,tanimoto,distance,target):
    mixture,log_length,log_noise=parameters
    length=np.exp(log_length);noise=np.exp(log_noise)
    rbf=np.exp(-.5*distance/length**2)
    covariance=mixture*tanimoto+(1-mixture)*rbf+(noise**2+1e-8)*np.eye(len(target))
    factor=cho_factor(covariance,lower=True)
    alpha=cho_solve(factor,target)
    value=.5*target@alpha+np.log(np.diag(factor[0])).sum()+.5*len(target)*np.log(2*np.pi)
    q=cho_solve(factor,np.eye(len(target)))-np.outer(alpha,alpha)
    gradient=np.array([.5*np.sum(q*(tanimoto-rbf)),
        .5*np.sum(q*(1-mixture)*rbf*distance/length**2),noise**2*np.trace(q)])
    return float(value),gradient


def fitted_posterior(train_x,train_y,test_x):
    mean=train_y.mean(0);scale=np.maximum(train_y.std(0),.1)
    standardized=(train_y-mean)/scale
    tanimoto,distance=components(train_x,train_x)
    cross_tanimoto,cross_distance=components(test_x,train_x)
    start=np.array([.7,0.,np.log(.05)])
    bounds=[(.05,.95),(np.log(.25),np.log(4.)),(np.log(.02),np.log(.3))]
    predicted=np.empty((len(test_x),train_y.shape[1]));std=np.empty_like(predicted);fits=[]
    for objective in range(train_y.shape[1]):
        target=standardized[:,objective]
        initial=likelihood(start,tanimoto,distance,target)[0]
        fit=minimize(likelihood,start,args=(tanimoto,distance,target),jac=True,
            method='L-BFGS-B',bounds=bounds,options={'maxiter':40,'ftol':1e-9})
        if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
            raise FloatingPointError('Nonfinite objective-specific GP fit')
        parameters=fit.x if fit.fun<=initial else start
        mixture,log_length,log_noise=parameters;length=np.exp(log_length);noise=np.exp(log_noise)
        covariance=mixture*tanimoto+(1-mixture)*np.exp(-.5*distance/length**2)
        covariance+=(noise**2+1e-8)*np.eye(len(target))
        factor=cho_factor(covariance,lower=True)
        cross=mixture*cross_tanimoto+(1-mixture)*np.exp(-.5*cross_distance/length**2)
        predicted[:,objective]=mean[objective]+cross@cho_solve(factor,target)*scale[objective]
        variance=np.maximum(1.-np.sum(cross*cho_solve(factor,cross.T).T,axis=1),1e-10)
        std[:,objective]=np.sqrt(variance)*scale[objective]
        fits.append(dict(tanimoto_weight=float(mixture),descriptor_length=float(length),
            noise=float(noise),converged=bool(fit.success),iterations=int(fit.nit),
            initial_nll=float(initial),fitted_nll=float(min(fit.fun,initial))))
    if not np.isfinite(predicted).all() or not np.isfinite(std).all():
        raise FloatingPointError('Nonfinite objective-specific GP prediction')
    return predicted,std,fits
